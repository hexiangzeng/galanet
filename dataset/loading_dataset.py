# -*- coding: utf-8 -*-
"""
# @Time    : May/23/2020
# @Author  : zhx
"""

from collections import OrderedDict
from batchgenerators.augmentations.utils import random_crop_2D_image_batched, pad_nd_image
import numpy as np
from batchgenerators.dataloading import SlimDataLoaderBase
from multiprocessing import Pool
from paths import preprocessed_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
default_num_threads = 8


def get_case_identifiers_from_raw_folder(folder):
    #  folder
    #    |--    caseid_0000.nii.gz
    case_identifiers = np.unique([i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz")])
    return case_identifiers


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def save_as_npz(args):
    if not isinstance(args, tuple):
        key = "data"
        npy_file = args
    else:
        npy_file, key = args
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def unpack_dataset(folder, threads=default_num_threads, key="data"):
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key]*len(npz_files)))
    p.close()
    p.join()


def pack_dataset(folder, threads=default_num_threads, key="data"):
    p = Pool(threads)
    npy_files = subfiles(folder, True, None, ".npy", True)
    p.map(save_as_npz, zip(npy_files, [key]*len(npy_files)))
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = subfiles(folder, join=False, suffix=".npz")
    npy_files = [join(folder, i+".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if isfile(i)]
    for n in npy_files:
        os.remove(n)


def load_dataset(folder, include_list=None):
    # we don't load the actual data but instead return the filename to the np file. the properties are loaded though
    case_identifiers = [s[:-4] for s in subfiles(folder, join=False, suffix=".npz")]
    if include_list is not None:
        case_identifiers = [c for c in case_identifiers if c in include_list]
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz"%c)
        dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl"%c))
    return dataset

