# -*- coding: utf-8 -*-
"""
# @Time    : Jun/16/2020
# @Author  : zhx
"""

import sys
from copy import deepcopy
import numpy as np
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from preprocessing.Preprocessor import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from batchgenerators.utilities.file_and_folder_operations import *


def save_segmentation_nifti_from_softmax(segmentation_softmax, out_fname, dct, order=1, region_class_order=None,
                                         seg_postprogess_fn=None, seg_postprocess_args=None, resampled_npz_fname=None,
                                         non_postprocessed_fname=None, force_separate_z=None,
                                         interpolation_order_z=0):
    print("force_separate_z:", force_separate_z, "interpolation order:", order)
    if isinstance(segmentation_softmax, str):
        assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        segmentation_softmax = np.load(segmentation_softmax)
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = dct.get('size_after_cropping')
    shape_original_before_cropping = dct.get('original_size_of_raw_data')
    print("current_shape:{}, shape_original_after_cropping:{}, shape_original_before_cropping:{}\n".format(current_shape,
                                                                                                         shape_original_after_cropping,
                                                                                                         shape_original_before_cropping))
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(dct.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(dct.get('original_spacing'))
            elif get_do_separate_z(dct.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(dct.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(dct.get('original_spacing'))
            else:
                lowres_axis = None

        print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z, cval=0,
                                               order_z=interpolation_order_z)
        #seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        save_pickle(dct, resampled_npz_fname[:-4] + ".pkl")

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = dct.get('lung_crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(dct['itk_spacing'])
    seg_resized_itk.SetOrigin(dct['itk_origin'])
    seg_resized_itk.SetDirection(dct['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(dct['itk_spacing'])
        seg_resized_itk.SetOrigin(dct['itk_origin'])
        seg_resized_itk.SetDirection(dct['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)


def save_segmentation_nifti(segmentation, out_fname, dct, order=1, force_separate_z=None):
    # suppress output
    print("force_separate_z:", force_separate_z, "interpolation order:", order)
    sys.stdout = open(os.devnull, 'w')

    if isinstance(segmentation, str):
        assert isfile(segmentation), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation)
        segmentation = np.load(segmentation)
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation.shape
    shape_original_after_cropping = dct.get('size_after_cropping')
    shape_original_before_cropping = dct.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any(np.array(current_shape) != np.array(shape_original_after_cropping)):
        if order == 0:
            seg_old_spacing = resize_segmentation(segmentation, shape_original_after_cropping, 0, 0)
        else:
            if force_separate_z is None:
                if get_do_separate_z(dct.get('original_spacing')):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(dct.get('original_spacing'))
                elif get_do_separate_z(dct.get('spacing_after_resampling')):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(dct.get('spacing_after_resampling'))
                else:
                    do_separate_z = False
                    lowres_axis = None
            else:
                do_separate_z = force_separate_z
                if do_separate_z:
                    lowres_axis = get_lowres_axis(dct.get('original_spacing'))
                else:
                    lowres_axis = None

            print("separate z:",do_separate_z, "lowres axis", lowres_axis)
            seg_old_spacing = resample_data_or_seg(segmentation[None], shape_original_after_cropping, is_seg=True,
                                                   axis=lowres_axis, order=order, do_separate_z=do_separate_z, cval=0)[0]
    else:
        seg_old_spacing = segmentation

    bbox = dct.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
    seg_resized_itk.SetSpacing(dct['itk_spacing'])
    seg_resized_itk.SetOrigin(dct['itk_origin'])
    seg_resized_itk.SetDirection(dct['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    sys.stdout = sys.__stdout__