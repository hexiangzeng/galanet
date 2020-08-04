# -*- coding: utf-8 -*-
"""
# @Time    : Jun/05/2020
# @Author  : zhx
"""

import torch


def maybe_to_torch(data):
    if isinstance(data, list):
        data = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in data]
    elif not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    return data


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.contiguous()
        data = data.cuda(gpu_id, non_blocking=True)
    return data
