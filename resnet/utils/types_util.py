"""
Types util.
"""

from typing import Union

import torch as tc


Device = Union[str, int]
Module = tc.nn.Module
Optimizer = tc.optim.Optimizer
Scaler = tc.cuda.amp.GradScaler
Scheduler = Union[tc.optim.lr_scheduler._LRScheduler, tc.optim.lr_scheduler.ReduceLROnPlateau]
Checkpointable = Union[Module, Optimizer, Scheduler]
Dataset = Union[tc.utils.data.Dataset, tc.utils.data.IterableDataset]
Sampler = tc.utils.data.DistributedSampler
Dataloader = tc.utils.data.DataLoader