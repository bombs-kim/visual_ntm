import argparse
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import NTM
from data import CopyDataset
from utils import update_monitored_state


def clip_grads(model, args):
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data.clamp_(args.min_grad, args.max_grad)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sequence_length', type=int, default=3)
    parser.add_argument('--sequence_width', type=int, default=10)
    parser.add_argument('--num_memory_locations', type=int, default=64)
    parser.add_argument('--memory_vector_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_size', type=int, default=999999)
    parser.add_argument('--controller_hidden_size', type=int, default=512)
    parser.add_argument('--controller_output_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_grad', type=float, default=-10.)
    parser.add_argument('--max_grad', type=float, default=10.)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save', type=str, default='backup')
    parser.add_argument('--monitor_state', action='store_true')

    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = CopyDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)

    model = NTM(N=args.num_memory_locations,
                M=args.memory_vector_size,
                in_seq_width=args.sequence_width,
                out_seq_width=args.sequence_width,
                ctr_hidden_size=args.controller_hidden_size,
                ctr_out_size=args.controller_output_size,
                monitor_state=args.monitor_state)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    if args.load != '':
        model.load_state_dict(torch.load(args.load))

    best = 999999
    losses = []
    for idx, (x, y) in enumerate(dataloader):
        if model.monitor_state:
            update_monitored_state(memory=None, read_head=None, write_head=None)

        model.reset_state(args.batch_size)
        optimizer.zero_grad()
        if model.monitor_state:
            update_monitored_state(*model.get_memory_info())

        seq_len = args.sequence_length

        for t in range(seq_len+2):
            model(x[:, t])

        pred = []
        for i in range(seq_len):
            pred.append(model())
        pred = torch.stack(pred, dim=1)

        loss = criterion(pred, y)
        loss.backward()
        # clip_grads(model, args)
        optimizer.step()
        losses.append(loss.item())

        if idx % 50 == 0:
            mean_loss = np.array(losses[:20]).mean()
            print("\n%8d" % idx, "Loss: %0.5f" % loss.item(),
                  "Mean: %0.5f" % mean_loss)
            losses = []
            torch.save(model.state_dict(), args.save)
            if mean_loss < best and loss.item() < best * 1.2:
                print("** best", idx, mean_loss)
                best = mean_loss
                torch.save(model.state_dict(), 'best')

if __name__ == "__main__":
    main()
