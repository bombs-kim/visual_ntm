import json
import os
import time

import torch


def update_monitored_state(memory=None, read_head=None,
                           write_head=None, filename='data.json'):
    if os.path.exists('data.json'):
        with open(filename) as f:
            try:
                j = json.load(f)
            except json.decoder.JSONDecodeError:
                j = {}
    else:
        j = {}

    for key, val in (('memory', memory), ('read_head', read_head),
                     ('write_head', write_head)):
        key_prev = key + '_prev'
        j[key_prev] = j[key] if key in j else None
        if val is not None:
            # read/write heads
            if isinstance(val, list):
                value = [v.clone().detach() for v in val][-10:]
                value += [torch.zeros(value[0].shape)] * (10 - len(value))
                value = torch.cat(value, 0).numpy().tolist()
            else:
                value = val.clone().detach().numpy().tolist()
        else:
            value = None
        j[key] = value

    with open(filename, 'w') as f:
        json.dump(j, f)

    time.sleep(0.3)
