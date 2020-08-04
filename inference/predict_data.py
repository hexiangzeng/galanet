# -*- coding: utf-8 -*-
"""
# @Time    : Jun/5/2020
# @Author  : zhx
"""

import torch
import numpy as np
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
from batchgenerators.augmentations.utils import pad_nd_image
from utilites.to_torch import maybe_to_torch, to_cuda
from utilites.softmax_helper import softmax_torch
from utilites.tensor_utilites import flip_tensor


def get_device(net):
    if next(net.parameters()).device == "cpu":
        return "cpu"
    else:
        return next(net.parameters()).device.index


def set_device(net, device):
    if device == "cpu":
        net.cpu()
    else:
        net.cuda(device)


def predict_3D(net, x, do_mirroring=True, num_repeats=1, use_train_mode=False, batch_size=1,
               mirror_axes=(0,1,2), tiled=True, tile_in_z=True, step=2, patch_size=None,
               use_gaussian=True, regions_class_order=None,
               pad_border_mode="edge", pad_kwargs=None, all_in_gpu=False, num_classes=None, return_softmax=True):
    print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)
    assert get_device(net) != "cpu", "CPU not implemented"
    current_mode = net.training
    if use_train_mode is not None and use_train_mode: # 蒙特卡洛采样
        raise RuntimeError("use_train_mode=True is currently broken! @Fabian needs to fix this "
                           "(don't put batchnorm layer into train, just dropout)")
        net.train()
    elif use_train_mode is not None and not use_train_mode:
        net.eval()
    else: pass
    assert len(x.shape) == 4, "data must have shape (c,d,h,w)"
    if tiled:
        res = _internal_predict_3D_tiled(net, x, num_repeats, batch_size, tile_in_z, step, do_mirroring,
                                         mirror_axes, patch_size, regions_class_order, use_gaussian,
                                         pad_border_mode, pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                         num_classes=num_classes, return_softmax=return_softmax)
    else:
        res = _internal_predict_3D(net, x, do_mirroring, num_repeats, patch_size, batch_size,
                                   mirror_axes, regions_class_order, pad_border_mode,
                                   pad_kwargs=pad_kwargs)
    if use_train_mode is not None:
        net.train(current_mode)
    return res


_gaussian_3d = None
_patch_size_for_gaussian_3d = None
def _internal_predict_3D_tiled(net, x, num_repeats=1, BATCH_SIZE=None, tile_in_z=True, step=2,
                               do_mirroring=True, mirror_axes=(0, 1, 2), patch_size=None,
                               regions_class_order=None, use_gaussian=False, pad_border_mode="constant",
                               pad_kwargs=None, all_in_gpu=False, num_classes=None, return_softmax=True):
    assert len(x.shape) == 4, "x must be (c, x, y, z)"
    print("step:", step)
    print("do mirror:", do_mirroring)

    torch.cuda.empty_cache()

    with torch.no_grad():
        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)  # x (1,303,265,372)

        data = data[None]  # data shape (1,1,303,265,372)

        if BATCH_SIZE is not None:
            data = np.vstack([data] * BATCH_SIZE)

        # input_size = [1, x.shape[0]] + list(patch_size)
        if not tile_in_z:
            # input_size[2] = data.shape[2]
            patch_size[0] = data.shape[2]
        # input_size = [int(i) for i in input_size]

        if num_classes is None:
            num_classes = net.num_classes

        if use_gaussian:
            global _gaussian_3d, _patch_size_for_gaussian_3d
            if _gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, _patch_size_for_gaussian_3d)]):
                tmp = np.zeros(patch_size)
                center_coords = [i // 2 for i in patch_size]
                sigmas = [i / 8 for i in patch_size]
                tmp[tuple(center_coords)] = 1
                tmp_smooth = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
                tmp_smooth = tmp_smooth / tmp_smooth.max() * 1
                add = tmp_smooth  # + 1e-7

                # tmp_smooth cannot be 0, otherwise we may end up with nans!
                tmp_smooth[tmp_smooth == 0] = np.min(tmp_smooth[tmp_smooth != 0])

                # we only need to compute that once. It can take a while to compute this due to the large sigma in
                # gaussian_filter
                _gaussian_3d = np.copy(add)
                _patch_size_for_gaussian_3d = deepcopy(patch_size)
            else:
                print("using precomputed Gaussian")
                add = _gaussian_3d
        else:
            add = np.ones(patch_size, dtype=np.float32)

        add = add.astype(np.float32)

        data_shape = data.shape

        print("configuring tiles")
        center_coord_start = np.array([i // 2 for i in patch_size]).astype(int)
        center_coord_end = np.array(
            [data_shape[i + 2] - patch_size[i] // 2 for i in range(len(patch_size))]).astype(int)
        num_steps = np.ceil(
            [(center_coord_end[i] - center_coord_start[i]) / (patch_size[i] / step) for i in range(3)])
        step_size = np.array(
            [(center_coord_end[i] - center_coord_start[i]) / (num_steps[i] + 1e-8) for i in range(3)])
        step_size[step_size == 0] = 9999999
        xsteps = np.round(np.arange(center_coord_start[0], center_coord_end[0] + 1e-8, step_size[0])).astype(int)
        ysteps = np.round(np.arange(center_coord_start[1], center_coord_end[1] + 1e-8, step_size[1])).astype(int)
        zsteps = np.round(np.arange(center_coord_start[2], center_coord_end[2] + 1e-8, step_size[2])).astype(int)

        if all_in_gpu:
            print("initializing result array (on GPU)")
            # some of these can remain in half. We just need the reuslts for softmax so it won't hurt at all to reduce
            # precision. Inference is of course done in float
            result = torch.zeros([num_classes] + list(data.shape[2:]), dtype=torch.half, device=get_device(net))
            print("moving data to GPU")
            data = torch.from_numpy(data).cuda(get_device(net), non_blocking=True)
            print("initializing result_numsamples (on GPU)")
            result_numsamples = torch.zeros([num_classes] + list(data.shape[2:]), dtype=torch.half,
                                            device=get_device(net))
            print("moving add to GPU")
            # add = torch.from_numpy(add).cuda(get_device(net), non_blocking=True).half()
            add = torch.from_numpy(add).cuda(get_device(net), non_blocking=True)
            add_torch = add
        else:
            result = np.zeros([num_classes] + list(data.shape[2:]), dtype=np.float32)
            result_numsamples = np.zeros([num_classes] + list(data.shape[2:]), dtype=np.float32)
            add_torch = torch.from_numpy(add).cuda(get_device(net), non_blocking=True)

        print("data shape:", data_shape)
        print("patch size:", patch_size)
        print("steps (x, y, and z):", xsteps, ysteps, zsteps)
        print("number of tiles:", len(xsteps) * len(ysteps) * len(zsteps))
        # data, result and add_torch and result_numsamples are now on GPU
        for x in xsteps:
            lb_x = x - patch_size[0] // 2
            ub_x = x + patch_size[0] // 2
            for y in ysteps:
                lb_y = y - patch_size[1] // 2
                ub_y = y + patch_size[1] // 2
                for z in zsteps:
                    lb_z = z - patch_size[2] // 2
                    ub_z = z + patch_size[2] // 2

                    predicted_patch = \
                        _internal_maybe_mirror_and_pred_3D(net, data[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z],
                                                           num_repeats, num_classes, mirror_axes, do_mirroring, add_torch)[0]
                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    result[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch

                    if all_in_gpu:
                        result_numsamples[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add.half()
                    else:
                        result_numsamples[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add

        slicer = tuple(
            [slice(0, result.shape[i]) for i in range(len(result.shape) - (len(slicer) - 1))] + slicer[1:])
        result = result[slicer]
        result_numsamples = result_numsamples[slicer]

        softmax_pred = result / result_numsamples  # C,D,H,W

        # patient_data = patient_data[:, :old_shape[0], :old_shape[1], :old_shape[2]]
        if regions_class_order is None:
            predicted_segmentation = softmax_pred.argmax(0)
        else:
            softmax_pred_here = softmax_pred
            predicted_segmentation_shp = softmax_pred_here[0].shape
            predicted_segmentation = np.zeros(predicted_segmentation_shp, dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[softmax_pred_here[i] > 0.5] = c
        if all_in_gpu:
            print("copying results to CPU")
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            if return_softmax:
                softmax_pred = softmax_pred.half().detach().cpu().numpy()
            else:
                softmax_pred = None

    print("prediction on GPU done")
    return predicted_segmentation, softmax_pred


def _intenal_predict_3D(net, x, do_mirroring, num_repeats, patch_size=None, batch_size=None,
                        mirror_axes=(0, 1, 2), regions_class_order=None, pad_border_mode="edge", pad_kwargs=None):
    pass


def _internal_maybe_mirror_and_pred_3D(net, x, num_repeats, num_classes, mirror_axes, do_mirroring=True, mult=None,
                                       inference_apply_nonlin=softmax_torch):
    with torch.no_grad():
        x = to_cuda(maybe_to_torch(x), gpu_id=get_device(net))
        result_torch = torch.zeros([1, num_classes] + list(x.shape[2:]),
                                   dtype=torch.float).cuda(get_device(net), non_blocking=True)
        mult = to_cuda(maybe_to_torch(mult), gpu_id=get_device(net))

        num_results = num_repeats
        if do_mirroring:
            mirror_idx = 8
            num_results *= 2 ** len(mirror_axes)
        else:
            mirror_idx = 1

        for i in range(num_repeats):
            for m in range(mirror_idx):
                if m == 0:
                    pred = inference_apply_nonlin(net(x))
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    pred = inference_apply_nonlin(net(flip_tensor(x, 4)))
                    result_torch += 1 / num_results * flip_tensor(pred, 4)

                if m == 2 and (1 in mirror_axes):
                    pred = inference_apply_nonlin(net(flip_tensor(x, 3)))
                    result_torch += 1 / num_results * flip_tensor(pred, 3)

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    pred = inference_apply_nonlin(net(flip_tensor(flip_tensor(x, 4), 3)))
                    result_torch += 1 / num_results * flip_tensor(flip_tensor(pred, 4), 3)

                if m == 4 and (0 in mirror_axes):
                    pred = inference_apply_nonlin(net(flip_tensor(x, 2)))
                    result_torch += 1 / num_results * flip_tensor(pred, 2)

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    pred = inference_apply_nonlin(net(flip_tensor(flip_tensor(x, 4), 2)))
                    result_torch += 1 / num_results * flip_tensor(flip_tensor(pred, 4), 2)

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    pred = inference_apply_nonlin(net(flip_tensor(flip_tensor(x, 3), 2)))
                    result_torch += 1 / num_results * flip_tensor(flip_tensor(pred, 3), 2)

                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    pred = inference_apply_nonlin(net(flip_tensor(flip_tensor(flip_tensor(x, 3), 2), 4)))
                    result_torch += 1 / num_results * flip_tensor(flip_tensor(flip_tensor(pred, 3), 2), 4)

        if mult is not None:
            result_torch[:, :] *= mult

    return result_torch


def predict_preprocessed_data_return_softmax(net, data, do_mirroring=True, num_repeats=1,
                                             use_train_mode=False, batch_size=1,
                                             mirror_axes=(0,1,2), tiled=True, tile_in_z=True, step=2,
                                             patch_size=None, use_gaussian=True, all_in_gpu=False):
    return predict_3D(net, data, do_mirroring, num_repeats, use_train_mode, batch_size,
                      mirror_axes, tiled, tile_in_z, step, patch_size, use_gaussian=use_gaussian,
                      pad_border_mode="constant", pad_kwargs={'constant_values': 0},
                      all_in_gpu=all_in_gpu, num_classes=None)[1]
