"""
Checkpoint util.
"""

from typing import Optional, Dict, Any
import os
import re
import abc
import importlib

import torch as tc

from resnet.utils.types_util import Checkpointable


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
    latest_steps = _latest_n_checkpoint_steps(base_path, n=1, kind=kind)
    return latest_steps[-1] if len(latest_steps) > 0 else None


def _clean(base_path, kind, n=5):
    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=n, kind=kind)
    for fname in os.listdir(base_path):
        parsed = _parse_name(fname)
        if parsed['kind'] == kind and parsed['steps'] not in latest_n_steps:
            os.remove(os.path.join(base_path, fname))


def maybe_load_checkpoint(
        checkpoint_dir: str,
        kind_name: str,
        checkpointable: Checkpointable,
        map_location: str,
        steps: Optional[int]
) -> int:
    base_path = checkpoint_dir
    os.makedirs(base_path, exist_ok=True)
    steps_ = _latest_step(base_path, kind_name) if steps is None else steps
    path = os.path.join(base_path, _format_name(kind_name, steps_, 'pth'))
    if not os.path.exists(path):
        print(f"Bad {kind_name} checkpoint or none at {base_path} with step {steps}.")
        print("Running from scratch.")
        return 0
    state_dict = tc.load(path, map_location=map_location)
    checkpointable.load_state_dict(state_dict)
    print(f"Loaded {kind_name} checkpoint from {base_path}, with step {steps_}."),
    print("Continuing from checkpoint.")
    return steps_


def save_checkpoint(
        checkpoint_dir: str,
        kind_name: str,
        checkpointable: Checkpointable,
        steps: int
) -> None:
    base_path = checkpoint_dir
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, _format_name(kind_name, steps, 'pth'))
    state_dict = checkpointable.state_dict()
    tc.save(state_dict, path)
    _clean(base_path, kind_name, n=5)


def maybe_load_checkpoints(
        checkpoint_dir: str,
        checkpointables: Dict[str, Optional[Checkpointable]],
        map_location: str,
        steps: Optional[int]
) -> int:
    """
    :param checkpoint_dir: Checkpoint dir.
    :param checkpointables: Dictionary of checkpointables keyed by kind name.
    :param map_location: Map location specifying how remap storage locations.
    :param steps: Number of steps so far. If None, uses latest.
    :return: Number of steps in latest checkpoint. If no checkpoints, returns 0.
    """
    global_steps = list()
    for kind_name in checkpointables:
        checkpointable = checkpointables.get(kind_name)
        if checkpointable is not None:
            step_ = maybe_load_checkpoint(
                checkpoint_dir=checkpoint_dir,
                kind_name=kind_name,
                checkpointable=checkpointable,
                map_location=map_location,
                steps=steps)
            global_steps.append(step_)
    if len(set(global_steps)) != 1:
        msg = "Checkpoint steps not aligned."
        raise RuntimeError(msg)
    return step_


def save_checkpoints(
        checkpoint_dir: str,
        checkpointables: Dict[str, Optional[Checkpointable]],
        steps: int
) -> None:
    """
    :param checkpoint_dir: Checkpoint dir.
    :param checkpointables: Dictionary of checkpointables keyed by kind name.
    :param rank: Process rank.
    :param steps: Number of steps so far.
    :return: None
    """
    for kind_name in checkpointables:
        checkpointable = checkpointables.get(kind_name)
        if checkpointable is not None:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                kind_name=kind_name,
                checkpointable=checkpointable,
                steps=steps)


class CheckpointStrategy(tc.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, unit):
        assert unit in ['batch', 'epoch']
        super().__init__()
        self._unit = unit
        self.register_buffer('_batch_step', tc.tensor(0))
        self.register_buffer('_epoch_step', tc.tensor(0))

    @property
    def unit(self):
        return self._unit

    @property
    def batch_step(self):
        return self._batch_step.item()

    @property
    def epoch_step(self):
        return self._epoch_step.item()

    def step(self, unit):
        assert unit in ['batch', 'epoch']
        if unit == 'batch':
            self.register_buffer('_batch_step', tc.tensor(self.batch_step+1))
        if unit == 'epoch':
            self.register_buffer('_epoch_step', tc.tensor(self.epoch_step+1))

    @abc.abstractmethod
    def observe(self, **kwargs) -> bool:
        """
        Observe the inputs, update state, and return checkpoint eligibility.
        """
        pass


class FrequencyCheckpointStrategy(CheckpointStrategy):
    def __init__(self, unit, frequency, **kwargs):
        super().__init__(unit)
        self._frequency = frequency

    def observe(self, unit, **kwargs) -> bool:
        cond = getattr(self, f"{unit}_step") % self._frequency == 0
        self.step(unit)
        if self.unit == unit:
            return cond
        return False


class PerformanceCheckpointStrategy(CheckpointStrategy):
    def __init__(self, unit, **kwargs):
        super().__init__(unit)
        self.register_buffer('_lowest_loss', tc.tensor(float('inf')))

    @property
    def lowest_loss(self):
        return self._lowest_loss.item()

    def observe(self, unit, loss, **kwargs) -> bool:
        cond = loss < self.lowest_loss
        self.step(unit)
        if self.unit == unit:
            if cond:
                self.register_buffer('_lowest_loss', tc.tensor(loss))
            return cond
        return False


def get_checkpoint_strategy(
        checkpoint_strategy_cls_name: str,
        checkpoint_strategy_args: Optional[Dict[str, Any]]
) -> CheckpointStrategy:
    """
    :param checkpoint_strategy_cls_name: CheckpointStrategy class name.
    :param checkpoint_strategy_args: Checkpoint strategy args.
    :return:
    """
    if checkpoint_strategy_args is None:
        checkpoint_strategy_args = dict()
    module = importlib.import_module('resnet.utils.checkpoint_util')
    cls = getattr(module, checkpoint_strategy_cls_name)
    return cls(**checkpoint_strategy_args)
