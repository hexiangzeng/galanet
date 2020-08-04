# -*- coding: utf-8 -*-
"""
# @Time    : May/17/2020
# @Author  : zhx
"""

import os
import numpy as np
import SimpleITK as sitk
from paths import *
from utilites.utils import crop
from preprocessing.DatasetAnalyzer import DatasetAnalyzer
from preprocessing.Preprocessor import Preprocessor
from batchgenerators.utilities.file_and_folder_operations import *
import os

# 将lung lobe数据集进行预处理
# 1) split data  # traing dataset/ test dataset
# 2) data crop: dicom_data -> 对肺区域crop，保存至copped_data  #
#              基于全局阈值的肺实质分割
# 3) dataset analyze:
#                     collect info: npz(volume数据和segmentation), pkl(当前病人原始数据包含信息，如size, spacings, class),
#                     dataset.json(训练集，测试集数据ids和位置), dataset_properties(整个数据集的all_sizes, all_spacings),
#                     intensity_properties(可选)
# 4) data normlize, 统一spacing


class RunPreprocessing():
    def __init__(self, num_threads, raw_data_dir=raw_cropped_data_dir, preprocessed_dir=preprocessed_output_dir):
        self.num_threads = num_threads
        self.raw_data_dir = raw_data_dir
        self.preprocessed_dir = preprocessed_dir

    def preprocessing(self, overwrite_cropped=False, analyse=False, run_preprocessing=False):
        self.crop(overwrite_cropped)
        self.dataset_analyse(analyse)
        if run_preprocessing or len(os.listdir(self.preprocessed_dir)) == 0:
            pp = Preprocessor(self.raw_data_dir, self.preprocessed_dir)
            pp.run()

    def crop(self, overwrite_cropped=False):
        crop(overwrite_cropped, self.raw_data_dir, num_threads=self.num_threads)

    def dataset_analyse(self, analyse=False):
        dataset_analyzer = DatasetAnalyzer(raw_cropped_data_dir, overwrite=False)  # this class creates the fingerprint
        if analyse or not os.path.isfile(dataset_analyzer.dataset_properties_file):
            if not os.path.isfile(dataset_analyzer.dataset_properties_file):
                print("dataset analyser must be executed at once.")
            dataset_analyzer.analyze_dataset()  # this will write output files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing Stage: crop lung area and analyse data.")
    parser.add_argument('-tf', type=int, required=False, default=8, help="threads to preprocess")
    parser.add_argument('--verify_dataset', required=False, default=False, action='store_true',
                        help="check dataset integrity!")
    args = parser.parse_args()
    tf = args.tf
    if args.verify_dataset:
        pass
    RunPreprocessing(tf).preprocessing(False, False, True)


if __name__ == '__main__':
    main()
