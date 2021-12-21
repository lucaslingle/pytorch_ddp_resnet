"""
Learning rate scheduler util.
"""

import importlib


def get_scheduler(scheduler_cls_name, optimizer, scheduler_args):
    module = importlib.import_module('torch.optim.lr_scheduler')
    scheduler_cls = getattr(module, scheduler_cls_name)
    return scheduler_cls(optimizer, **scheduler_args)
