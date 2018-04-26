import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: mini-batch mode support


# Becuase Head doensn't have any parameters,
# it doesn't make much sense to inherit torch.nn.Module.

class Head:
    def __init__(self, N, M, shift_range=3):
        self.N, self.M = N, M
        self.shift_range = shift_range
        self.in_features = self.M + self.shift_range + 3
        self.reset_state()

    def reset_state(self):
        self.w_prev = torch.zeros((1, self.N), dtype=torch.float32)
        self.history = []  # history is only used for debug purpose

    # ********************** Addressing Implementation ***********************
    # Following methods have been named after Fig.2
    def content_addressing(self, mem, key, stren):
        w_tmp = F.cosine_similarity(mem, key.view(1, -1), 1, 1e-16)
        w_c = F.softmax(stren * w_tmp, dim=-1)
        return w_c

    def interpolation(self, w_c, w_prev, gate):
        w_g = gate * w_c + (1-gate) * w_prev
        return w_g

    def convolutional_shift(self, w_g, shift):
        w_tmp = torch.cat([w_g[:, -1:], w_g, w_g[:, :1]], dim=1)
        w_tmp = w_tmp.unsqueeze(1)  # resulting shape (Batch, 1, self.N)
        shift = shift.unsqueeze(1)
        # Maybe I need to call contiguous here..
        w_shifted = F.conv1d(w_tmp, shift).squeeze(1)
        return w_shifted

    def sharpening(self, w_shifted, sharp):
        w_tmp = w_shifted ** sharp
        w = w_tmp / (torch.sum(w_tmp) + 1e-16)
        return w
    # ************************************************************************

    def __call__(self, controller_outputs, memory):
        assert controller_outputs.shape[1] == self.in_features
        # key  : k, key vector for content-based addressing
        # stren: β, key strength scalar
        # gate : g, interpolation gate scalar
        # shift: s, shift weighting vector
        # sharp: γ, sharpening scalar

        key = controller_outputs[:, :self.M]
        other_params = controller_outputs[:, self.M:]

        stren, gate, sharp = torch.split(other_params[:, :3], (1, 1, 1), dim=1)
        shift = other_params[:, 3:]
        assert shift.shape[1] == self.shift_range

        w_c = self.content_addressing(memory, key, stren)
        w_g = self.interpolation(w_c, self.w_prev, gate)
        w_shifted = self.convolutional_shift(w_g, shift)
        w = self.sharpening(w_shifted, sharp)

        self.w_prev = w
        return w


class Controller(nn.Module):
    # TODO: Make hidden_features_1, hidden_features_2 configurable
    def __init__(self, in_features, out_features,
                 hidden_features_1=256, hidden_features_2=512):
        super().__init__()
        # 얘네 구조는 논문 그대로는 아니라고 밝히기 ㅠ
        self.fc1 = nn.Linear(in_features, hidden_features_1)
        self.fc2 = nn.Linear(hidden_features_1, hidden_features_2)
        self.fc3 = nn.Linear(hidden_features_2, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.4)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.4)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1.4)
        nn.init.normal_(self.fc1.bias, std=0.01)
        nn.init.normal_(self.fc2.bias, std=0.01)
        nn.init.normal_(self.fc3.bias, std=0.01)

    def forward(self, x, prev_read):
        x = torch.cat((x, prev_read), dim=1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # Skip activaiton for the final layer
        x = torch.sigmoid(self.fc3(x))
        return x


class NTM(nn.Module):
    def __init__(self, N, M, in_seq_width, out_seq_width, shift_range=3):
        super().__init__()
        self.N, self.M = N, M
        self.in_seq_width = in_seq_width

        # TODO: support multiple heads for read/write
        # Assuming read_head.in_features == write_head.in_features
        self.read_head = Head(N, M, shift_range=shift_range)
        self.write_head = Head(N, M, shift_range=shift_range)

        self.head_in = self.read_head.in_features
        self.controller = Controller(
            self.in_seq_width + M,
            self.head_in * 2 + M * 2)  # M * 2: erase and add in size

        self.fc = nn.Linear(self.in_seq_width + M, out_seq_width)
        self.memory = torch.zeros((N, M), dtype=torch.float32)
        # self.prev_read = torch.zeros((1, N), dtype=torch.float32)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.4)
        nn.init.normal_(self.fc.bias, std=0.5)

    def reset_state(self):
        stdev = 1 / (np.sqrt(self.N + self.M))
        self.memory = nn.init.uniform(
            torch.Tensor(self.N, self.M), -stdev, stdev)
        self.prev_read = torch.tanh(torch.randn(1, self.M, dtype=torch.float32))
        self.read_head.reset_state()
        self.write_head.reset_state()

    def read(self, controller_outputs):
        w = self.read_head(controller_outputs, self.memory)
        return w @ self.memory

    def write(self, controller_outputs):
        outputs, a, e = torch.split(
            controller_outputs, (self.head_in, self.M, self.M), dim=1)
        w = self.write_head(outputs, self.memory)

        # Assuming batch size is always 1
        w = torch.squeeze(w)
        e = torch.squeeze(e)
        a = torch.squeeze(a)
        erase = torch.ger(w, e)
        add = torch.ger(w, a)

        # element-wise multiplication
        mem = self.memory * (1 - erase)
        mem = mem + add
        self.memory = mem

    def forward(self, x=None):
        # import ipdb
        # ipdb.set_trace(context=9)
        if x is None:
            assert hasattr(self, 'batch_size')
            x = torch.zeros(self.batch_size, self.in_seq_width)
        self.batch_size = x.shape[0]

        controller_outputs = self.controller(x, self.prev_read)
        routputs = controller_outputs[:, :self.head_in]
        woutputs = controller_outputs[:, self.head_in:]

        # Should I clone this?
        self.prev_read = read_out = self.read(routputs)
        self.write(woutputs)

        x = torch.cat((x, read_out), 1)
        x = torch.sigmoid(self.fc(x))
        return x
