# -*- coding: utf-8 -*-
"""
# @Time    : May/26/2020
# @Author  : zhx
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import numpy as np

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def net_summary(self, net):

        x = torch.rand(1, 1, 64, 140, 140).cuda()
        self.writer.add_graph(net, (x,))

    def image_summary(self, tag, images:torch.Tensor, step):
        """Log a list of images."""
        images = normlize_watch_image(images)  # must normlize to 0-1 firstly
        if len(images.shape) == 4:  # images shape  c, d, h, w
            if images.shape[0] != 1:
                images = images[:1]
            # images = torch.stack([images[0], images[0], images[0]]).transpose(0, 1)
            # images = make_grid(images, padding=10, normalize=True,
            #                    scale_each=True, pad_value=1)
            images = torch.stack([images, images, images], dim=1)
            self.writer.add_images(tag, images, step, dataformats='NCHW')
        elif len(images.shape) == 3:
            id = np.random.choice(images.shape[0], 1).item()
            images = torch.cat([images[id], images[id], images[id]], dim=0)
            self.writer.add_image(tag, images, step, dataformats='CHW')
        elif len(images.shape) == 2: # h, w
            images = torch.stack([images, images, images], dim=0)
            self.writer.add_image(tag, images, step, dataformats='CHW')
        else:
            print("summary images error. tag:{}, image shape:{}, step:{}".format(tag, images.shape, step))


def normlize_watch_image(tensor):
    tensor = tensor.clone()  # avoid modifying tensor in-place

    minv = float(tensor.min())
    maxv = float(tensor.max())
    tensor.clamp_(min=minv, max=maxv)
    tensor.add_(-minv).div_(maxv - minv + 1e-5)

    return tensor


if __name__ == '__main__':
    tensors = torch.randint(0, 10, size=(1, 3, 3, 3))
    print(float(tensors.min()), float(tensors.max()))
    tensors_normlizes = normlize_watch_image(tensors)
    print(float(tensors_normlizes.min()), float(tensors_normlizes.max()))
