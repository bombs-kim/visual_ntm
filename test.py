import argparse
from time import time, sleep

import numpy as np
import torch

from model import NTM
from data import CopyDataset
from utils import update_monitored_state


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--sequence_width', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--monitor_state', action='store_true')
    return parser.parse_args()


def pprint(tensor, ndigits=2):
    tensor = tensor[0]  # remove batch
    tensor = torch.round(tensor * (10**ndigits)) / (10**ndigits)
    template = '%0.' + str(ndigits) + 'f'
    for line in tensor:
        print(', '.join(template % n for n in line))


def main():
    torch.set_printoptions(precision=1, linewidth=240)
    args = parse_arguments()
    _, _, model_state, init_args, _ = torch.load(args.load)
    init_args[-1] = args.monitor_state
    model = NTM(*init_args)
    model.load_state_dict(model_state)

    dataset = CopyDataset(
        args.sequence_length, args.sequence_width, 100, args.batch_size)

    for idx, (x, delimeter, y) in enumerate(dataset):
        if model.monitor_state:
            update_monitored_state(
                memory=None, read_head=None, write_head=None)

        model.reset_state(args.batch_size)
        if model.monitor_state:
            update_monitored_state(*model.get_memory_info())

        seq_len = x.shape[1]
        for t in range(seq_len):
            model(x[:, t])
        model(delimeter)
        pred = []
        for i in range(seq_len):
            pred.append(model())
        pred = torch.stack(pred, dim=1)

        print('pred')
        pprint(pred)
        print('y')
        pprint(y)
        print()
        sleep(1)

if __name__ == "__main__":
    main()
