# -*- coding: utf-8 -*-
"""
# @Time    : Jun/04/2020
# @Author  : zhx
"""

# -*- coding: utf-8 -*-
"""
# @Time    : Jul/17/2020
# @Author  : zhx
"""

import os
import argparse
import torch
import shutil
from paths import *
import numpy as np
from copy import deepcopy
from network_architecture import Unet, GALANet, GALANetv2, APLSNet, PDVNet, update_lr, \
    DiceLossCrossEntropy, dice_coff, DiceLossFocalLoss, SoftDiceLoss, load_best_checkpoint, load_latest_checkpoint
from preprocessing.main_run_preprocessing import RunPreprocessing
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool


def check_input_folder_and_return_caseIDs(input_folder):
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-7] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        expected_output_file = c + ".nii.gz"
        if not isfile(join(input_folder, expected_output_file)):
            missing.append(expected_output_file)
        else:
            remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids), np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining), np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def predict_cases(model, list_of_lists, output_filenames, folds, save_npz, num_threads_preprocessing,
                  num_threads_nifti_save, overwrite_existing=False, all_in_gpu=False, step=2, force_separate_z=None,
                  interp_order=3, interp_order_z=0, checkpoint_name="model_best"):
    assert len(list_of_lists) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    debug_info = load_json(join(model, str(folds), "debug.json"))
    network_name = os.path.basename(model)
    assert network_name in ("3DUNet", "GALANet", "APLSNet", "PDVNet")
    in_channels = 1
    num_classes = 5
    if network_name == "3DUNet":
        net = Unet(in_channels, num_classes+1, patch_size=[128, 160, 160], features_num_lists=[24, 48, 96, 192, 384]).cuda()
    elif network_name == "GALANet":
        net = GALANet(in_channels, num_classes+1, patch_size=[128, 192, 192], features_num_lists=[16, 32, 64, 128]).cuda()
    elif network_name == "APLSNet":
        net = APLSNet(in_channels, num_classes+1, patch_size=[128, 192, 192], features_num_lists=[32, 40, 48, 56]).cuda()
    elif network_name == "PDVNet":
        net = PDVNet(in_channels, num_classes+1, patch_size=[128, 192, 192], base_features=24).cuda()
    else:
        raise "{} error!!!".format(network_name)
    if checkpoint_name.endswith("best"):
        net, optimizer, epoch = load_best_checkpoint(join(model, "fold_{}".format(folds)), net, optimizer=None, train=False)
    else:
        net, optimizer, epoch = load_latest_checkpoint(join(model, "fold_{}".format(folds)), net, optimizer=None, train=False)

    print("starting preprocessing")
    tmp_crop_folder = join(os.path.dirname(output_filenames), "tmp_crop_folder")

    print("starting prediction...")

    print("inference done. Now waiting for the segmentation export to finish...")
    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning


def predict_from_folder(model, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, overwrite_existing=True, overwrite_all_in_gpu=None, step=2,
                        force_separate_z=None, interp_order=3, interp_order_z=0):
    maybe_mkdir_p(output_folder)

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(input_folder)

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 7)] for j in case_ids]

    if overwrite_all_in_gpu is None:
        all_in_gpu = False
    else:
        all_in_gpu = overwrite_all_in_gpu

    return predict_cases(model, list_of_lists, output_files, folds,
                         save_npz, num_threads_preprocessing, num_threads_nifti_save,
                         overwrite_existing=overwrite_existing, all_in_gpu=all_in_gpu, step=step,
                         force_separate_z=force_separate_z, interp_order=interp_order, interp_order_z=interp_order_z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-m', '--model', help="GALANet, 3DUNet, PLSNet, PDVNet",
                        default="GALANet", required=False)
    parser.add_argument('-f', '--folds', nargs='+', default=1,
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="use this if you want to ensemble these predictions with those of other models. Softmax "
                             "probabilities will be saved as compressed numpy arrays in output_folder and can be "
                             "merged between output_folders with nnUNet_ensemble_predictions")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
                        "Determines many background processes will be used for data preprocessing. Reduce this if you "
                        "run into out of memory (RAM) problems. Default: 6")
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
                        "Determines many background processes will be used for segmentation export. Reduce this if you "
                        "run into out of memory (RAM) problems. Default: 2")
    parser.add_argument("--interp_order", required=False, default=3, type=int,
                        help="order of interpolation for segmentations, has no effect if mode=fastest. Do not touch this.")
    parser.add_argument("--interp_order_z", required=False, default=0, type=int,
                        help="order of interpolation along z is z is done differently. Do not touch this.")
    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. "
                                                                                       "Do not touch.")
    args = parser.parse_args("-i '/media/zhx/My Passport/lung_lobe_seg/galaNet_raw_data/splited_data/test_images' "
                             "-o '/media/zhx/My Passport/lung_lobe_seg/galaNet_raw_data/splited_data/GALANet_test_prediction' "
                             "-m GALANet -f 1".split())
    input_folder = args.input_folder
    output_folder = args.output_folder
    folds = args.folds
    save_npz = args.save_npz
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    step = args.step
    interp_order = args.interp_order
    interp_order_z = args.interp_order_z
    force_separate_z = None
    overwrite_existing = args.overwrite_existing
    all_in_gpu = args.all_in_gpu
    model = args.model
    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False
    model_folder_name = join(network_output_dir_base, model)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save,
                        overwrite_existing=overwrite_existing, overwrite_all_in_gpu=all_in_gpu,
                        step=step, force_separate_z=force_separate_z, interp_order=interp_order,
                        interp_order_z=interp_order_z)
