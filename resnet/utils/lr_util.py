"""
Learning rate scheduler util.
"""

from typing import Optional, Dict, Any
import importlib

import torch as tc


def get_scheduler(
        scheduler_cls_name: Optional[str],
        optimizer: tc.optim.Optimizer,
        scheduler_args: Dict[str, Any]
):
    if scheduler_cls_name == 'None':
        return None
    module = importlib.import_module('torch.optim.lr_scheduler')
    scheduler_cls = getattr(module, scheduler_cls_name)
    return scheduler_cls(optimizer, **scheduler_args)
