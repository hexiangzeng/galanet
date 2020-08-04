# -*- coding: utf-8 -*-
"""
# @Time    : May/29/2020
# @Author  : zhx
"""

from time import time
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from utilites.utils import print_to_log_file
from network_architecture.Unet import Unet
import torch.optim as optim
from paths import network_output_dir_base
from collections import OrderedDict


def save_checkpoint(epoch, net, optimizer, fname, save_optimizer=True):
    start_time = time()
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    if save_optimizer:
        optimizer_state_dict = optimizer.state_dict()
    else:
        optimizer_state_dict = None

    fold = fname.split(os.sep)[-2][-1]
    network_name = fname.split(os.sep)[-3]
    print_to_log_file("saving checkpoint...", network=network_name, fold=fold)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer_state_dict': optimizer_state_dict},
        fname)
    print_to_log_file("done, saving took %.2f seconds" % (time() - start_time), network=network_name, fold=fold)


def load_latest_checkpoint(output_folder, net, optimizer, train=True):
    print(output_folder)
    if isfile(join(output_folder, "model_final_checkpoint.model")):
        return load_checkpoint(join(output_folder, "model_final_checkpoint.model"), net, optimizer, train=train)
    if isfile(join(output_folder, "model_latest.model")):
        return load_checkpoint(join(output_folder, "model_latest.model"), net, optimizer, train=train)
    if isfile(join(output_folder, "model_best.model")):
        return load_best_checkpoint(output_folder, net, optimizer, train)
    raise RuntimeError("No checkpoint found")


def load_best_checkpoint(output_folder, net, optimizer, train=True):
    if isfile(join(output_folder, "model_best.model")):
        return load_checkpoint(join(output_folder, "model_best.model"), net, optimizer, train=train)
    else:
        fold = output_folder.split(os.sep)[-1][-1]
        network_name = output_folder.split(os.sep)[-2]
        print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                          "back to load_latest_checkpoint", network=network_name, fold=fold)
        return load_latest_checkpoint(train, net, optimizer)


def load_checkpoint(fname, net, optimizer=None, train=True):
    fold = fname.split(os.sep)[-2][-1]
    network_name = fname.split(os.sep)[-3]
    print_to_log_file("loading checkpoint", fname, "train=", train, network=network_name, fold=fold)
    # net initialized optimizer
    # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
    saved_model = torch.load(fname, map_location=torch.device('cpu'))

    new_state_dict = OrderedDict()
    curr_state_dict_keys = list(net.state_dict().keys())
    # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in saved_model['state_dict'].items():
        key = k
        if key not in curr_state_dict_keys:
            print("duh")
            key = key[7:]  # erase 'module.'
        new_state_dict[key] = value
    net.load_state_dict(new_state_dict)
    epoch = saved_model['epoch']
    if train:
        optimizer_state_dict = saved_model['optimizer_state_dict']
        if optimizer is None:
            optimizer = optim.SGD(net.parameters(), lr=0.01)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
    print_to_log_file("loading checkpoint done.\n", network=network_name, fold=fold)
    return net, optimizer, epoch


if __name__ == '__main__':
    net = Unet(1, 6, 2, [16, 32, 64, 128, 128])
    optimizer = optim.Adam(net.parameters(), lr=0.01)
