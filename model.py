import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import update_monitored_state


# TODO: mini-batch mode support

class Head(nn.Module):
    def __init__(self, N, M, in_size, shift_range=3):
        super().__init__()
        self.N, self.M = N, M
        self.shift_range = shift_range
        self.fc = nn.Linear(in_size, self.M + self.shift_range + 3)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.4)
        nn.init.normal_(self.fc.bias, std=0.01)

    def reset_state(self, batch_size):
        self.w_prev = torch.zeros((batch_size, self.N), dtype=torch.float)
        self.history = [self.w_prev]  # history is only used for debug purpose

    # ###################### Addressing Implementation ######################
    # Following methods were named after Fig.2 of NTM - Graves et al.
    def content_addressing(self, mem, key, stren):
        w_tmp = F.cosine_similarity(mem, key.unsqueeze(1), dim=2, eps=1e-16)
        w_c = F.softmax(stren * w_tmp, dim=-1)
        return w_c

    def interpolation(self, w_c, w_prev, gate):
        w_g = gate * w_c + (1-gate) * w_prev
        return w_g

    def convolutional_shift(self, w_g, shift):
        w_g = torch.cat([w_g[:, -1:], w_g, w_g[:, :1]], dim=1)
        w_list = []
        for w_each, shift_each in zip(w_g, shift):
            #            resulting shape (Batch, channel, self.N)
            w_each = w_each.reshape(1, 1, -1)
            #            shift weight shape (out_channel, in_channel, range)
            shift_each = shift_each.reshape(1, 1, -1)
            w_shifted_each = F.conv1d(w_each, shift_each).squeeze(1)
            w_list.append(w_shifted_each)

        w_shifted = torch.cat(w_list, dim=0)
        return w_shifted

    def sharpening(self, w_shifted, sharp):
        w_tmp = w_shifted ** sharp
        w = w_tmp / (torch.sum(w_tmp) + 1e-16)
        return w
    # ###
    # #######################################################################

    def forward(self, controller_outputs, memory):
        outputs = self.fc(controller_outputs)

        # key  : k, key vector for content-based addressing
        # stren: β, key strength scalar
        # gate : g, interpolation gate scalar
        # shift: s, shift weighting vector
        # sharp: γ, sharpening scalar

        key = outputs[:, :self.M]
        other_params = outputs[:, self.M:]

        stren, gate, sharp = torch.split(other_params[:, :3], (1, 1, 1), dim=1)
        shift = other_params[:, 3:]

        assert shift.shape[1] == self.shift_range

        key = torch.tanh(key)
        stren = F.softplus(stren)
        gate = torch.sigmoid(gate)
        shift = torch.softmax(shift, dim=-1)
        sharp = 1 + F.softplus(sharp)  # TODO: check

        w_c = self.content_addressing(memory, key, stren)
        w_g = self.interpolation(w_c, self.w_prev, gate)
        w_shifted = self.convolutional_shift(w_g, shift)
        w = self.sharpening(w_shifted, sharp)

        self.w_prev = w
        self.history.append(w)
        return w


class Controller(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.4)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.4)
        nn.init.normal_(self.fc1.bias, std=0.01)
        nn.init.normal_(self.fc2.bias, std=0.01)

    def forward(self, x, prev_read):
        x = torch.cat((x, prev_read), dim=1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class LstmController(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, out_size)
        # TODO: add self.reset_parameters()

    def reset_state(self, batch_size):
        shape = (self.num_layers, batch_size, self.hidden_size)
        self.hidden = (torch.zeros(shape), torch.zeros(shape))

    def forward(self, x, prev_read):
        x = torch.cat((x, prev_read), dim=1)

        # Add sequence dimension
        x = x.unsqueeze(dim=0)
        x, self.hidden = self.lstm(x, self.hidden)

        x = x.squeeze(dim=0)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class NTM(nn.Module):
    def __init__(self, N, M, in_seq_width, out_seq_width,
                 ctr_hidden_size, ctr_out_size,
                 shift_range=3, monitor_state=False):
        super().__init__()
        self.N, self.M = N, M
        self.in_seq_width = in_seq_width
        self.monitor_state = monitor_state

        self.controller = LstmController(
            self.in_seq_width + M, ctr_hidden_size, ctr_out_size)

        # TODO: support multiple heads for read/write
        self.read_head = Head(N, M, ctr_out_size, shift_range=shift_range)
        self.write_head = Head(N, M, ctr_out_size, shift_range=shift_range)

        self.erase_add_fc = nn.Linear(256, M * 2)
        self.fc = nn.Linear(self.in_seq_width + M, out_seq_width)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.erase_add_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.4)
        nn.init.normal_(self.erase_add_fc.bias, std=0.5)
        nn.init.normal_(self.fc.bias, std=0.5)

    def reset_state(self, batch_size):
        self.batch_size = batch_size

        self.controller.reset_state(batch_size)

        # TODO: Maybe make prev_read zero?
        self.prev_read = torch.tanh(
            torch.randn(batch_size, self.M, dtype=torch.float))
        # TODO: add batchsize to read_head, write_head reset_state
        self.read_head.reset_state(batch_size)
        self.write_head.reset_state(batch_size)

        stdev = 1 / (np.sqrt(self.N + self.M))
        self.memory = nn.init.uniform(
            torch.Tensor(batch_size, self.N, self.M), -stdev, stdev)

    def read(self, controller_outputs):
        w = self.read_head(controller_outputs, self.memory)
        w = w.unsqueeze(1)
        return torch.bmm(w, self.memory).squeeze(1)

    def write(self, controller_outputs):
        w = self.write_head(controller_outputs, self.memory)

        ea = self.erase_add_fc(controller_outputs)
        e, a = ea[:, :self.M], ea[:, self.M:]
        e = torch.sigmoid(e)
        a = torch.tanh(a)

        w = w.unsqueeze(2)  # shape (batch, self.N, 1)
        e = e.unsqueeze(1)  # shape (batch, 1, self.M)
        a = a.unsqueeze(1)

        erase = torch.bmm(w, e)  # shape (batch, self.N, self.M)
        add = torch.bmm(w, a)
        mem = self.memory * (1 - erase)
        mem = mem + add
        self.memory = mem

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.in_seq_width)

        controller_outputs = self.controller(x, self.prev_read)

        # TODO: need to clone this?
        self.prev_read = read_out = self.read(controller_outputs)
        if self.monitor_state:
            update_monitored_state(*self.get_memory_info())
        self.write(controller_outputs)
        if self.monitor_state:
            update_monitored_state(*self.get_memory_info())

        x = torch.cat((x, read_out), 1)
        x = torch.sigmoid(self.fc(x))
        return x

    def get_memory_info(self):
        return self.memory, self.read_head.history, self.write_head.history
