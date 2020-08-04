# -*- coding: utf-8 -*-
"""
# @Time    : Jun/19/2020
# @Author  : zhx
"""

import numpy as np
import random
from paths import preprocessed_output_dir, preprocessed_data_identifer
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    get_range_val, gaussian_filter
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy

default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,

    "do_elastic": False,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",

    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,

    "num_threads": 12,
    "num_cached_per_thread": 1,
}


def remove_seg_label(seg, remove_label, replace_with=0):
    seg[seg == remove_label] = replace_with
    return seg


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)


def augment_rot90(sample_data, sample_seg, num_rot=(1, 2, 3), axes=(0, 1, 2)):
    """
    :param sample_data:
    :param sample_seg:
    :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
    :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
    :return:
    """
    num_rot = np.random.choice(num_rot)
    axes = np.random.choice(axes, size=2, replace=False)
    axes.sort()
    axes = [i + 1 for i in axes]
    sample_data = np.rot90(sample_data, num_rot, axes)
    if sample_seg is not None:
        sample_seg = np.rot90(sample_seg, num_rot, axes)
    return sample_data, sample_seg


def augment_resize(sample_data, sample_seg, target_size, order=3, order_seg=1, cval_seg=0):
    """
    Reshapes data (and seg) to target_size
    :param sample_data: np.ndarray or list/tuple of np.ndarrays, must be (c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param target_size: int or list/tuple of int
    :param order: interpolation order for data (see skimage.transform.resize)
    :param order_seg: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param sample_seg: can be None, if not None then it will also be resampled to target_size. Can also be list/tuple of
    np.ndarray (just like data). Must also be (c, x, y(, z))
    :return:
    """
    dimensionality = len(sample_data.shape) - 1
    if not isinstance(target_size, (list, tuple)):
        target_size_here = [target_size] * dimensionality
    else:
        assert len(target_size) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                   "the same dimensionality as data!"
        target_size_here = list(target_size)

    sample_data = resize_multichannel_image(sample_data, target_size_here, order)

    if sample_seg is not None:
        target_seg = np.ones([sample_seg.shape[0]] + target_size_here)
        for c in range(sample_seg.shape[0]):
            target_seg[c] = resize_segmentation(sample_seg[c], target_size_here, order_seg, cval_seg)
    else:
        target_seg = None

    return sample_data, target_seg


def augment_zoom(sample_data, sample_seg, zoom_factors, order=3, order_seg=1, cval_seg=0):
    """
    zooms data (and seg) by factor zoom_factors
    :param sample_data: np.ndarray or list/tuple of np.ndarrays, must be (c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param zoom_factors: int or list/tuple of int (multiplication factor for the input size)
    :param order: interpolation order for data (see skimage.transform.resize)
    :param order_seg: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param sample_seg: can be None, if not None then it will also be zoomed by zoom_factors. Can also be list/tuple of
    np.ndarray (just like data). Must also be (c, x, y(, z))
    :return:
    """

    dimensionality = len(sample_data.shape) - 1
    shape = np.array(sample_data.shape[1:])
    if not isinstance(zoom_factors, (list, tuple)):
        zoom_factors_here = np.array([zoom_factors] * dimensionality)
    else:
        assert len(zoom_factors) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                    "the same dimensionality as data!"
        zoom_factors_here = np.array(zoom_factors)
    target_shape_here = list(np.round(shape * zoom_factors_here).astype(int))

    sample_data = resize_multichannel_image(sample_data, target_shape_here, order)

    if sample_seg is not None:
        target_seg = np.ones([sample_seg.shape[0]] + target_shape_here)
        for c in range(sample_seg.shape[0]):
            target_seg[c] = resize_segmentation(sample_seg[c], target_shape_here, order_seg, cval_seg)
    else:
        target_seg = None

    return sample_data, target_seg


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if max(axes) > 2:
        raise ValueError("augment_mirroring now takes the axes as the spatial dimensions. What previously was "
                         "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                         "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg


def augment_channel_translation(data, const_channel=0, max_shifts=None):
    if max_shifts is None:
        max_shifts = {'z': 2, 'y': 2, 'x': 2}

    shape = data.shape

    const_data = data[:, [const_channel]]
    trans_data = data[:, [i for i in range(shape[1]) if i != const_channel]]

    # iterate the batch dimension
    for j in range(shape[0]):

        slice = trans_data[j]

        ixs = {}
        pad = {}

        if len(shape) == 5:
            dims = ['z', 'y', 'x']
        else:
            dims = ['y', 'x']

        # iterate the image dimensions, randomly draw shifts/translations
        for i, v in enumerate(dims):
            rand_shift = np.random.choice(list(range(-max_shifts[v], max_shifts[v], 1)))

            if rand_shift > 0:
                ixs[v] = {'lo': 0, 'hi': -rand_shift}
                pad[v] = {'lo': rand_shift, 'hi': 0}
            else:
                ixs[v] = {'lo': abs(rand_shift), 'hi': shape[2 + i]}
                pad[v] = {'lo': 0, 'hi': abs(rand_shift)}

        # shift and pad so as to retain the original image shape
        if len(shape) == 5:
            slice = slice[:, ixs['z']['lo']:ixs['z']['hi'], ixs['y']['lo']:ixs['y']['hi'],
                    ixs['x']['lo']:ixs['x']['hi']]
            slice = np.pad(slice, ((0, 0), (pad['z']['lo'], pad['z']['hi']), (pad['y']['lo'], pad['y']['hi']),
                                   (pad['x']['lo'], pad['x']['hi'])),
                           mode='constant', constant_values=(0, 0))
        if len(shape) == 4:
            slice = slice[:, ixs['y']['lo']:ixs['y']['hi'], ixs['x']['lo']:ixs['x']['hi']]
            slice = np.pad(slice, ((0, 0), (pad['y']['lo'], pad['y']['hi']), (pad['x']['lo'], pad['x']['hi'])),
                           mode='constant', constant_values=(0, 0))

        trans_data[j] = slice

    data_return = np.concatenate([const_data, trans_data], axis=1)
    return data_return


def augment_spatial_transform(data, seg=None, patch_size=None, patch_center_dist_from_border=None,
                              do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.), do_rotation=True,
                              angle_x=default_3D_augmentation_params.get("rotation_x"),
                              angle_y=default_3D_augmentation_params.get("rotation_y"),
                              angle_z=default_3D_augmentation_params.get("rotation_z"),
                              do_scale=True, scale=(0.85, 1.25),
                              border_mode_data='constant', border_cval_data=0, order_data=3,
                              border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
                              random_crop=False, p_el_per_sample=1,
                              p_scale_per_sample=1, p_rot_per_sample=1):
    if len(data.shape) == 3:
        print("Attention: data shape must be (C, D, H, W). I will transform it.")
        data = data[None]
        if seg is not None and len(seg.shape) == 3:
            seg = seg[None]
    if patch_size is None:
        patch_size = (data.shape[1], data.shape[2], data.shape[3])

    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    coords = create_zero_centered_coordinate_mesh(patch_size)
    modified_coords = False
    if np.random.uniform() < p_el_per_sample and do_elastic_deform:
        a = np.random.uniform(alpha[0], alpha[1])
        s = np.random.uniform(sigma[0], sigma[1])
        coords = elastic_deform_coordinates(coords, a, s)
        modified_coords = True

    if np.random.uniform() < p_rot_per_sample and do_rotation:
        if angle_x[0] == angle_x[1]:
            a_x = angle_x[0]
        else:
            a_x = np.random.uniform(angle_x[0], angle_x[1])
        if dim == 3:
            if angle_y[0] == angle_y[1]:
                a_y = angle_y[0]
            else:
                a_y = np.random.uniform(angle_y[0], angle_y[1])
            if angle_z[0] == angle_z[1]:
                a_z = angle_z[0]
            else:
                a_z = np.random.uniform(angle_z[0], angle_z[1])
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)
        else:
            coords = rotate_coords_2d(coords, a_x)
        modified_coords = True

    if np.random.uniform() < p_scale_per_sample and do_scale:
        if np.random.random() < 0.5 and scale[0] < 1:
            sc = np.random.uniform(scale[0], 1)
        else:
            sc = np.random.uniform(max(scale[0], 1), scale[1])
        coords = scale_coords(coords, sc)
        modified_coords = True

    # now find a nice center location
    if modified_coords:
        for d in range(dim):
            if random_crop:
                ctr = np.random.uniform(patch_center_dist_from_border[d],
                                        data.shape[d + 1] - patch_center_dist_from_border[d])
            else:
                ctr = int(np.around(data.shape[d + 1] / 2.))
            coords[d] += ctr
        for channel_id in range(data.shape[0]):
            data_result[channel_id] = interpolate_img(data[channel_id], coords, order_data,
                                                      border_mode_data, cval=border_cval_data)
        if seg is not None:
            for channel_id in range(seg.shape[0]):
                seg_result[channel_id] = interpolate_img(seg[channel_id], coords, order_seg,
                                                         border_mode_seg, cval=border_cval_seg, is_seg=True)
    else:
        if seg is None:
            s = None
        else:
            s = seg[None]
        if random_crop:
            margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
            d, s = random_crop_aug(data[None], s, patch_size, margin)
        else:
            d, s = center_crop_aug(data[None], patch_size, s)
        data_result = d[0]
        if seg is not None:
            seg_result = s[0]
    return data_result, seg_result


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1), p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


def augment_gaussian_blur(data_sample, blur_sigma_range=(1, 5), different_sigma_per_channel=True,
                          p_per_channel=1, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if not different_sigma_per_channel:
            sigma = get_range_val(blur_sigma_range)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                if different_sigma_per_channel:
                    sigma = get_range_val(blur_sigma_range)
                data_sample[c] = np.asarray(gaussian_filter(data_sample[c], sigma, order=0))
    return data_sample


def augment_brightness_multiplicative(data_sample, multiplier_range=(0.5, 2), per_channel=True, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
        if not per_channel:
            data_sample *= multiplier
        else:
            for c in range(data_sample.shape[0]):
                multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
                data_sample[c] *= multiplier
    return data_sample


def augment_brightness_additive(data_sample, mu, sigma, per_channel=True, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if not per_channel:
            rnd_nb = np.random.normal(mu, sigma)
            data_sample += rnd_nb
        else:
            for c in range(data_sample.shape[0]):
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if not per_channel:
            mn = data_sample.mean()
            if preserve_range:
                minm = data_sample.min()
                maxm = data_sample.max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample = (data_sample - mn) * factor + mn
            if preserve_range:
                data_sample[data_sample < minm] = minm
                data_sample[data_sample > maxm] = maxm
        else:
            for c in range(data_sample.shape[0]):
                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()
                if np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                data_sample[c] = (data_sample[c] - mn) * factor + mn
                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


def augment_linear_downsampling(data_sample, zoom_range=(0.5, 1), per_channel=True, p_per_channel=1,
                                channels=None, order_downsample=1, order_upsample=0, ignore_axes=None, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        augment_linear_downsampling_scipy(data_sample, zoom_range=zoom_range, per_channel=per_channel,
                                          p_per_channel=p_per_channel, channels=channels, order_downsample=order_downsample,
                                          order_upsample=order_upsample, ignore_axes=ignore_axes)
    return data_sample


def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False,  per_channel=True, epsilon=1e-7,
                  retain_stats=False, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if invert_image:
            data_sample = - data_sample
        if not per_channel:
            if retain_stats:
                mn = data_sample.mean()
                sd = data_sample.std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample.min()
            rnge = data_sample.max() - minm
            data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            if retain_stats:
                data_sample = data_sample - data_sample.mean() + mn
                data_sample = data_sample / (data_sample.std() + 1e-8) * sd
        else:
            for c in range(data_sample.shape[0]):
                if retain_stats:
                    mn = data_sample[c].mean()
                    sd = data_sample[c].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data_sample[c].min()
                rnge = data_sample[c].max() - minm
                data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                if retain_stats:
                    data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                    data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
        if invert_image:
            data_sample = - data_sample
    return data_sample


def select_channel_from_seg(seg_sample, channels=[0]):
    assert isinstance(channels, (tuple, list)), "channels must be list or tuple."
    assert len(seg_sample.shape)==3 or len(seg_sample.shape)==4, "seg shape must like (c, h, w) or (c, d, h, w)."
    return seg_sample[channels]


def set_augment_parameters_for_test(params=default_3D_augmentation_params, patch_size=None):
    data_aug_params = params
    data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    data_aug_params["scale_range"] = (0.7, 1.4)
    data_aug_params["do_elastic"] = False
    data_aug_params['selected_seg_channels'] = [0]
    data_aug_params['patch_size_for_spatialtransform'] = patch_size
    return data_aug_params


if __name__ == '__main__':
    # basic_patch_size = get_patch_size((64, 160, 160),
    #                                   default_3D_augmentation_params['rotation_x'],
    #                                   default_3D_augmentation_params['rotation_y'],
    #                                   default_3D_augmentation_params['rotation_z'],
    #                                   default_3D_augmentation_params['scale_range'])
    # print(basic_patch_size)
    from batchgenerators.utilities.file_and_folder_operations import *
    from matplotlib import pyplot as plt
    from dataset.lungdataset import do_augmentation
    from time import time
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    slice_idx = 50
    preprocessed_dir = join(preprocessed_output_dir, preprocessed_data_identifer)
    files_path = subfiles(preprocessed_dir, suffix=".npy")[0]
    all_data = np.load(files_path)
    data = all_data[:-1]
    seg = all_data[-1:]
    # data = all_data[:-1][:, 50:100, 100:200, 100:200]  # c, d, h, w
    # seg = all_data[-1:][:, 50:100, 100:200, 100:200]  # c, d, h, w
    print("before transform shape:{}, {}, seg unique:{}".format(data.shape, seg.shape, np.unique(seg)))
    axes[0, 0].imshow(data[0][slice_idx], plt.cm.gray, interpolation='nearest')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(seg[0][slice_idx], plt.cm.gray, interpolation='nearest')
    axes[0, 1].axis('off')
    data_aug_params = set_augment_parameters_for_test()
    # basic_patch_size = get_patch_size((96, 192, 192),
    #                                   default_3D_augmentation_params['rotation_x'],
    #                                   default_3D_augmentation_params['rotation_y'],
    #                                   default_3D_augmentation_params['rotation_z'],
    #                                   default_3D_augmentation_params['scale_range'])
    # print(basic_patch_size)
    data_res, seg_res = do_augmentation(data, seg, params=data_aug_params, patch_size=(96, 192, 192))
    axes[1, 0].imshow(data_res[0][slice_idx], plt.cm.gray, interpolation='nearest')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(seg_res[0][slice_idx], plt.cm.gray, interpolation='nearest')
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()
    print("after transform shape:{}, {}, seg unique:{}".format(data_res.shape, seg_res.shape, np.unique(seg_res)))
