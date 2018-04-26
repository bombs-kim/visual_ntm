import numpy as np

import torch
from torch.utils.data import Dataset


class CopyDataset(Dataset):
    def __init__(self, args):
        self.seq_len = args.sequence_length
        self.seq_width = args.sequence_width
        self.training_size = args.training_size

    def create_seq(self):
        seq = np.random.binomial(1, 0.5, (self.seq_len, self.seq_width))
        seq = torch.from_numpy(seq)
        inp = torch.zeros(self.seq_len + 2, self.seq_width)
        inp[1:self.seq_len + 1, :self.seq_width] = seq.clone()
        inp[0, 0] = 1.0
        inp[self.seq_len + 1, self.seq_width - 1] = 1.0
        outp = seq.data.clone()
        return inp.float(), outp.float()

    def __len__(self):
        return self.training_size

    def __getitem__(self, _):
        inp, out = self.create_seq()
        return inp, out
