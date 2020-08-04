# -*- coding: utf-8 -*-
"""
# @Time    : May/25/2020
# @Author  : zhx
"""


def update_lr(optimizer, lr, epoch, max_epochs, exponent=0.9):
    """Sets the learning rate to a fixed number"""
    optimizer.param_groups[0]['lr'] = lr * (1 - epoch / max_epochs)**exponent
