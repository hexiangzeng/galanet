# -*- coding: utf-8 -*-
"""
# @Time    : May/18/2020
# @Author  : zhx
"""

import sys
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from paths import *
from preprocessing.cropper import ImageCropper
import importlib
import pkgutil
from time import time, sleep
from datetime import datetime
from paths import network_output_dir_base
default_num_threads = 8


def recursive_find_class(folder, net_class, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, net_class):
                tr = getattr(m, net_class)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_class([join(folder[0], modname)], net_class, current_module=next_current_module)
            if tr is not None:
                break

    return tr


def get_lists_from_splited_dir(base_split_dir):
    lists = []

    json_file = join(base_split_dir, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        cur_pat.append(join(raw_splited_dir, "train_images", tr['image'].split("/")[-1]))
        cur_pat.append(join(raw_splited_dir, "train_labels", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}


def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]


def crop(overwrite=False, crop_out_dir=raw_cropped_data_dir, num_threads=default_num_threads):
    if overwrite and isdir(crop_out_dir):
        shutil.rmtree(crop_out_dir)
        maybe_mkdir_p(crop_out_dir)
    case_lists_files, _ = get_lists_from_splited_dir(raw_splited_dir)

    imgcrop = ImageCropper(num_threads, crop_out_dir)
    imgcrop.run_cropping(case_lists_files, overwrite_existing=overwrite)
    shutil.copy(join(raw_splited_dir, "dataset.json"), crop_out_dir)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def print_to_log_file(*log_args, network=None, fold=1, also_print_to_console=True, add_timestamp=True):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        log_args = ("%s:" % dt_object, *log_args)

    log_file = subfiles(join(network_output_dir_base, network, "fold_%s" % str(fold)),
                        join=True, prefix="training_log", suffix=".txt")
    # assert len(log_file) == 1 or len(log_file) == 0, "one network one log file or zero file!"
    if len(log_file) == 1: log_file = log_file[0]
    else: log_file = None

    if log_file is None:
        output_folder = join(network_output_dir_base, network, "fold_%s" % str(fold))
        maybe_mkdir_p(output_folder)
        timestamp = datetime.now()
        log_file = join(output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                        (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                         timestamp.second))
        with open(log_file, 'w') as f:
            f.write("Starting... \n")
    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(log_file, 'a+') as f:
                for a in log_args:
                    f.write(str(a))
                    f.write("\t")
                f.write("\n")
            successful = True
        except IOError:
            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*log_args)


if __name__ == '__main__':
    print_to_log_file("saving checkpoint...")
