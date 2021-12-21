"""
Types util.
"""

from typing import Union

import torch as tc


Module = tc.nn.Module
Optimizer = tc.optim.Optimizer
Scheduler = Union[tc.optim.lr_scheduler._LRScheduler, tc.optim.lr_scheduler.ReduceLROnPlateau]
Checkpointable = Union[Module, Optimizer, Scheduler]
Dataloader = tc.utils.data.DataLoader