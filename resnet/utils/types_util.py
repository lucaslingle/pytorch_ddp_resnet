"""
Types util.
"""

from typing import Union

import torch as tc


Module = tc.nn.Module
Optimizer = tc.optim.Optimizer
Scheduler = tc.optim.lr_scheduler._LRScheduler
Checkpointable = Union[Module, Optimizer, Scheduler]
Dataloader = tc.utils.data.DataLoader