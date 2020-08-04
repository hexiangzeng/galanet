# -*- coding: utf-8 -*-
"""
# @Time    : Jun/16/2020
# @Author  : zhx
"""

import torch
import numpy as np


def softmax_torch(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def softmax_numpy(x):
    x_max = np.max(x, axis=1, keepdims=True).repeat(x.shape[1], axis=1)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=1, keepdims=True).repeat(x.shape[1], axis=1)


if __name__ == '__main__':
    x = np.arange(0, 10).reshape(2, 5)
    y = softmax_numpy(x)
    print(y)
    # print(np.sum(y, axis=1))
