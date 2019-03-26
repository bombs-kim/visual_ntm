import argparse

import numpy as np
import torch

from model import NTM
from data import CopyDataset


def clip_grads(model, args):
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data.clamp_(args.min_grad, args.max_grad)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--sequence_width', type=int, default=8)
    parser.add_argument('--num_memory_locations', type=int, default=128)
    parser.add_argument('--memory_vector_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--training_size', type=int, default=999999)
    parser.add_argument('--controller_output_size', type=int, default=100)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--min_grad', type=float, default=-10.)
    parser.add_argument('--max_grad', type=float, default=10.)
    parser.add_argument('--weight_decay', type=int, default=1e-4)

    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save', type=str, default='backup')
    parser.add_argument('--save_best', type=str, default='best')
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.batch_size != 1:
        print("Warning. NTM cannot be easily trained"
              " with batch size bigger than 1")
    if args.load == '':
        idx = 0
        best = 999999
        model = NTM(N=args.num_memory_locations,
                    M=args.memory_vector_size,
                    in_seq_width=args.sequence_width,
                    out_seq_width=args.sequence_width,
                    ctr_out_size=args.controller_output_size)
    else:
        idx, best, model_state, init_args, optim_state = torch.load(args.load)
        idx += 1
        print('******** continuing from idx', idx, 'best', best)
        model = NTM(*init_args)
        model.load_state_dict(model_state)

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.learning_rate,
        momentum=args.momentum, alpha=args.alpha,
        weight_decay=args.weight_decay)
    if args.load != '':
        optimizer.load_state_dict(optim_state)

    criterion = torch.nn.BCELoss()
    dataset = CopyDataset(args.sequence_length,
                          args.sequence_width,
                          args.training_size,
                          args.batch_size)
    losses = []
    for idx, (x, delimeter, y) in enumerate(dataset, idx):
        model.reset_state(args.batch_size)
        optimizer.zero_grad()

        seq_len = x.shape[1]
        for t in range(seq_len):
            model(x[:, t])
        model(delimeter)
        pred = []
        for i in range(seq_len):
            pred.append(model())
        pred = torch.stack(pred, dim=1)

        loss = criterion(pred, y)
        loss.backward()
        clip_grads(model, args)
        optimizer.step()
        losses.append(loss.item())

        if idx % 50 == 0:
            mean_loss = np.array(losses[:20]).mean()
            losses = []
            torch.save([idx, best, model.state_dict(), model.init_args,
                        optimizer.state_dict()], args.save)
            if idx != 0 and mean_loss < best and loss.item() < best * 1.5:
                print("******** best")
                best = mean_loss
                torch.save([idx, best, model.state_dict(), model.init_args,
                            optimizer.state_dict()], args.save_best)
            print("%8d" % idx, "Loss: %0.5f" % loss.item(),
                  "Recent average: %0.5f" % mean_loss)

if __name__ == "__main__":
    main()
