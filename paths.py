# -*- coding: utf-8 -*-
"""
# @Time    : May/16/2020
# @Author  : zhx
"""

import os
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

preprocessed_data_identifer = "data_preprocessed"
preprocessed_net_inputs = "net_inputs"

# raw_data_base_dir = "/media/zhx/My Passport/lung_lobe_seg/galaNet_raw_data" # 原始数据保存文件夹
# preprocessed_output_dir = "/media/zhx/My Passport/lung_lobe_seg/galaNet_preprocessed" # 预处理后数据存放处
# network_output_dir_base = "/media/zhx/My Passport/lung_lobe_seg/galaNet_trained_models" # 网络存放处

raw_data_base_dir = "/data/fox_cloud/data/hexiang/lung_lobe_seg/galaNet_raw_data" # 原始数据保存文件夹
preprocessed_output_dir = "/data/fox_cloud/data/hexiang/lung_lobe_seg/galaNet_preprocessed" # 预处理后数据存放处
network_output_dir_base = "/data/fox_cloud/data/hexiang/lung_lobe_seg/galaNet_trained_models" # 网络存放处

# raw_data_base_dir = "/data0/mzs/zhx/lung_lobe_seg/galaNet_raw_data" # 原始数据保存文件夹
# preprocessed_output_dir = "/data0/mzs/zhx/lung_lobe_seg/galaNet_preprocessed" # 预处理后数据存放处
# network_output_dir_base = "/data0/mzs/zhx/lung_lobe_seg/galaNet_trained_models" # 网络存放处

# raw_data_base_dir = "/home/zenghexiang/data/zenghexiang/lung_lobe_seg/galaNet_raw_data" # 原始数据保存文件夹
# preprocessed_output_dir = "/home/zenghexiang/data/zenghexiang/lung_lobe_seg/galaNet_preprocessed" # 预处理后数据存放处
# network_output_dir_base = "/home/zenghexiang/data/zenghexiang/lung_lobe_seg/galaNet_trained_models" # 网络存放处

if raw_data_base_dir is not None:
    raw_dicom_data_dir = join(raw_data_base_dir, "dicom_data")  # dicom原始数据存放文件夹
    raw_cropped_data_dir = join(raw_data_base_dir, "cropped_data") # 原始数据被crop后存放的文件夹
    raw_splited_dir = join(raw_data_base_dir, "splited_data")
    maybe_mkdir_p(raw_data_base_dir)
    maybe_mkdir_p(raw_cropped_data_dir)
else:
    raise AssertionError("Attention! raw_data_base_dir is not defined! Please set raw_data_base_dir in paths.py.")

if preprocessed_output_dir is not None:
    maybe_mkdir_p(preprocessed_output_dir)
    maybe_mkdir_p(join(preprocessed_output_dir, preprocessed_data_identifer))
    maybe_mkdir_p(join(preprocessed_output_dir, preprocessed_net_inputs))
else:
    raise AssertionError("Attention! preprocessed_output_dir is not defined! "
                         "Please set preprocessed_output_dir in paths.py.")

if network_output_dir_base is not None:
    maybe_mkdir_p(network_output_dir_base)
else:
    raise AssertionError("Attention! network_output_dir_base is not defined! "
                         "Please set network_output_dir_base in paths.py.")
