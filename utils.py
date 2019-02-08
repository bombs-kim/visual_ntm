import json
import os
import time

import torch


def update_monitored_state(memory=None, read_head=None,
                           write_head=None, filename='data.json'):
    """
    Only the state with respect to the first item of a batch is monitored
    """
    if os.path.exists('data.json'):
        with open(filename) as f:
            try:
                j = json.load(f)
            except json.decoder.JSONDecodeError:
                j = {}
    else:
        j = {}

    NUM_HEAD_HISTORY = 10
    for key, val in (('memory', memory), ('read_head', read_head),
                     ('write_head', write_head)):
        key_prev = key + '_prev'
        j[key_prev] = j[key] if key in j else None
        if val is not None:
            # read/write heads
            if isinstance(val, list):
                value = [v.clone().detach() for v in val][-NUM_HEAD_HISTORY:]
                value += [torch.zeros(value[0].shape)] * (NUM_HEAD_HISTORY - len(value))
                value = torch.cat(value, 0).numpy().tolist()
            else:
                value = val[0].clone().detach().numpy().tolist()
        else:
            value = None
        j[key] = value

    with open(filename, 'w') as f:
        json.dump(j, f)

    time.sleep(0.3)
