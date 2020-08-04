# -*- coding: utf-8 -*-
"""
# @Time    : May/23/2020
# @Author  : zhx
"""

from paths import *
import scipy
import numpy as np
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict
from preprocessing.DatasetAnalyzer import load_properties_of_cropped
import shutil
from scipy.ndimage.interpolation import map_coordinates

## resample
# def resample(volume, origin_spacing, new_spacing=(1,1,1)):
#     # Determine current pixel spacing
#     resize_factor = origin_spacing / new_spacing  #DHW
#     new_real_shape = volume.shape * resize_factor
#     new_shape = np.round(new_real_shape)
#     real_resize_factor = new_shape / volume.shape
#     new_spacing = origin_spacing / real_resize_factor
#
#     image = scipy.ndimage.interpolation.zoom(volume, real_resize_factor, mode='nearest')
#
#     return image, new_spacing


RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD=3

# def resample_data_or_seg(data, new_shape, is_seg, order=3, cval=0):
#     if data is None: return data
#     assert len(data.shape) == 4, "data must be (c, x, y, z)"
#     if is_seg:
#         resize_fn = resize_segmentation
#         kwargs = OrderedDict()
#     else:
#         resize_fn = resize
#         kwargs = {'mode': 'edge', 'anti_aliasing': False}
#     dtype_data = data.dtype
#     data = data.astype(float)
#     shape = np.array(data[0].shape)
#     new_shape = np.array(new_shape)
#     if np.any(shape != new_shape):
#         print("no separate z, order", order)
#         reshaped = []
#         for c in range(data.shape[0]):
#             reshaped.append(resize_fn(data[c], new_shape, order, cval=cval, **kwargs)[None])
#         reshaped_final_data = np.vstack(reshaped)
#         return reshaped_final_data.astype(dtype_data)
#     else:
#         print("no resampling necessary")
#         return data


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, cval=cval,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z, cval=cval,
                                                                   mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                cval=cval, mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, cval=cval, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=1, force_separate_z=False,
                     cval_data=0, cval_seg=-1, order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z, cval=cval_data,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, cval=cval_seg,
                                            order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped


class Preprocessor(object):
    def __init__(self, cropped_data_dir=raw_cropped_data_dir, preprocessed_dir=preprocessed_output_dir,
                 datafolder_identider=preprocessed_data_identifer):
        self.cropped_data_dir = cropped_data_dir
        self.preprocessed_dir = preprocessed_dir
        maybe_mkdir_p(self.preprocessed_dir)
        self.preprocessed_data_dir = join(self.preprocessed_dir, datafolder_identider)
        self.list_of_cropped_npz_files = subfiles(self.cropped_data_dir, True, None, ".npz", True)
        self.case_identifiers = [c.split("/")[-1][:-4] for c in self.list_of_cropped_npz_files]
        maybe_mkdir_p(self.preprocessed_data_dir)

        sizes = []
        spacings = []
        for c in self.case_identifiers:
            properties = load_properties_of_cropped(self.cropped_data_dir, c)
            sizes.append(properties["size_after_cropping"])
            spacings.append(properties["original_spacing"])
        self.sizes = sizes
        self.original_spacing = spacings
        self.target_spacing_percentile = 50
        self.target_spacing = self.get_target_spacing()
        self.intensityproperties_file = join(cropped_data_dir, "intensityproperties.pkl")
        self.intensityproperties = load_pickle(self.intensityproperties_file)

    def run(self):
        print("Initializing to run preprocessing")
        print("npz folder:", self.cropped_data_dir)
        print("output_folder:", self.preprocessed_data_dir)

        for case_identifier, size, origin_spacing in zip(self.case_identifiers, self.sizes, self.original_spacing):
            print("Preprocessing: ", case_identifier)
            data, seg, properties = self.load_cropped(self.cropped_data_dir, case_identifier)
            data, seg, properties = self.resample_and_normalize(data, origin_spacing, self.target_spacing, properties, seg)
            all_data = np.vstack((data, seg)).astype(np.float32)
            print("Saving: {}".format(join(self.preprocessed_data_dir, case_identifier+".npz")))
            np.savez_compressed(join(self.preprocessed_data_dir, case_identifier+".npz"), data=all_data)
            write_pickle(properties, join(self.preprocessed_data_dir, case_identifier+".pkl"))
        files_copyed = ["dataset.json", "dataset_properties.pkl", "intensityproperties.pkl", "props_per_case.pkl"]
        for f in files_copyed:
            try:
                shutil.copy(join(self.cropped_data_dir, f), join(self.preprocessed_dir, f))
            except Exception as e:
                print("Error during coping dataset infomation files:{}.Please check code again.\n".format(f))
                print(e)

    def get_target_spacing(self):
        target = np.percentile(np.vstack(self.original_spacing), self.target_spacing_percentile, 0)
        return target

    def resample_and_normalize(self, data, original_spacing, target_spacing, properties, seg=None, force_separate_z=None):
        assert data is not None or seg is not None
        if data is not None: assert len(data.shape) == 4, "data must be 4D c,d,h,w"
        if seg is not None: assert len(seg.shape) == 4, "data must be 4D c,d,h,w"
        before = {
            'spacing': properties["original_spacing"],
            'data.shape': data.shape
        }
        # remove nans
        data[np.isnan(data)] = 0
        new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * data[0].shape)).astype(int)
        # data_reshaped = resample_data_or_seg(data, new_shape, False, 3, cval=0)
        # seg_reshaped = resample_data_or_seg(seg, new_shape, True, 1, cval=-1)
        # modified
        original_spacing_transposed = np.array(properties["original_spacing"])
        data_reshaped, seg_reshaped = resample_patient(
            data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
            force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
            separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data_reshaped.shape
        }
        print("before:", before, "\nafter: ", after, "\n")
        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing


        # normalize
        for c in range(len(data_reshaped)):
            # clip to lb and ub from train data foreground and use foreground mn and sd from training data
            assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
            mean_intensity = self.intensityproperties[c]['mean']
            std_intensity = self.intensityproperties[c]['sd']
            lower_bound = self.intensityproperties[c]['percentile_00_5']
            upper_bound = self.intensityproperties[c]['percentile_99_5']
            data_reshaped[c] = np.clip(data_reshaped[c], lower_bound, upper_bound)
            data_reshaped[c] = (data_reshaped[c] - mean_intensity) / std_intensity

        return data_reshaped, seg_reshaped, properties


    @staticmethod
    def load_cropped(cropped_output_dir, case_identifier):
        all_data = np.load(os.path.join(cropped_output_dir, "%s.npz" % case_identifier))['data']
        data = all_data[:-1].astype(np.float32)
        seg = all_data[-1:]
        with open(os.path.join(cropped_output_dir, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return data, seg, properties


def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis
