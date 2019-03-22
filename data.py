import random

import numpy as np
import torch
from torch.utils.data import Dataset


class CopyDataset:
    def __init__(self, max_seq_len, seq_width, training_size, batch_size):
        self.max_seq_len = max_seq_len
        self.seq_width = seq_width
        self.training_size = training_size
        self.batch_size = batch_size

    def __len__(self):
        return self.training_size

    def get_seq(self, seq_len):
        seq = np.random.binomial(1, 0.5, (seq_len, self.seq_width))
        seq = torch.from_numpy(seq)
        out = seq.data.clone()
        inp, out = seq.float(), out.float()
        return inp, out

    def get_batch(self):
        seq_len = random.randint(1, self.max_seq_len)
        inp, out = [], []
        for _ in range(self.batch_size):
            i, o = self.get_seq(seq_len)
            inp.append(i)
            out.append(o)
        inp_batch = torch.stack(inp, dim=0)
        delimeter = torch.zeros(self.batch_size, self.seq_width) - 1
        out_batch = torch.stack(out, dim=0)
        return inp_batch, delimeter, out_batch

    def __getitem__(self, _):
        return self.get_batch()
