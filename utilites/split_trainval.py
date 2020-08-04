# -*- coding: utf-8 -*-
"""
# @Time    : May/23/2020
# @Author  : zhx
"""

import numpy as np
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold
from paths import preprocessed_output_dir, preprocessed_data_identifer


def make_split_file(preprocess_dir=preprocessed_output_dir):
    splits_file = join(preprocess_dir, "trainval_splits.pkl")
    if not isfile(splits_file):
        print("Creating new split...")
        splits = []
        all_keys_sorted = subfiles(join(preprocess_dir, preprocessed_data_identifer), False, suffix=".npz")
        all_keys_sorted = [k[:-4] for k in all_keys_sorted]
        kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
        for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            splits.append(OrderedDict())
            splits[-1]['train'] = train_keys
            splits[-1]['val'] = test_keys
        save_pickle(splits, splits_file)


def load_split(fold, preprocess_dir=preprocessed_output_dir):
    splits = load_pickle(join(preprocess_dir, "trainval_splits.pkl"))
    train_keys = splits[fold-1]['train']
    val_keys = splits[fold-1]['val']
    return train_keys, val_keys
