"""
Checkpoint util.
"""

from typing import Union, Optional
import os
import re

import torch as tc


def _format_name(kind, steps, suffix):
    filename = f"{kind}_{steps}.{suffix}"
    return filename


def _parse_name(filename):
    m = re.match(r"(\w+)_([0-9]+).([a-z]+)", filename)
    return {
        "kind": m.group(1),
        "steps": int(m.group(2)),
        "suffix": m.group(3)
    }


def _latest_n_checkpoint_steps(base_path, n=5, kind=''):
    ls = os.listdir(base_path)
    grep = [f for f in ls if _parse_name(f)['kind'].startswith(kind)]
    steps = set(map(lambda f: _parse_name(f)['steps'], grep))
    latest_steps = sorted(steps)
    latest_n = latest_steps[-n:]
    return latest_n


def _latest_step(base_path, kind=''):
    return _latest_n_checkpoint_steps(base_path, n=1, kind=kind)[-1]


def _clean(base_path, kind, n=5):
    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=n, kind=kind)
    for fname in os.listdir(base_path):
        parsed = _parse_name(fname)
        if parsed['kind'] == kind and parsed['steps'] not in latest_n_steps:
            os.remove(os.path.join(base_path, fname))


def maybe_load_checkpoint(
        checkpoint_dir: str,
        kind_name: str,
        checkpointable: Union[tc.nn.Module, tc.optim.Optimizer],
        steps: Optional[int]
):
    try:
        base_path = checkpoint_dir
        steps_ = _latest_step(base_path, kind_name) if steps is None else steps
        path = os.path.join(base_path, _format_name(kind_name, steps_, 'pth'))
        state_dict = tc.load(path)
    except FileNotFoundError:
        print(f"Bad {kind_name} checkpoint or none at {base_path} with step {steps}.")
        print("Running from scratch.")
        return 0

    checkpointable.load_state_dict(state_dict)
    print(f"Loaded {kind_name} checkpoint from {base_path}, with step {steps_}."),
    print("Continuing from checkpoint.")
    return steps_


def save_checkpoint(
        checkpoint_dir: str,
        kind_name: str,
        checkpointable: Union[tc.nn.Module, tc.optim.Optimizer],
        rank: int,
        steps: int
):
    if rank == 0:
        base_path = checkpoint_dir
        os.makedirs(base_path, exist_ok=True)
        path = os.path.join(base_path, _format_name(kind_name, steps, 'pth'))
        state_dict = checkpointable.state_dict()
        tc.save(state_dict, path)
        _clean(base_path, kind_name, n=5)
