# -*- coding: utf-8 -*-
"""
# @Time    : May/18/2020
# @Author  : zhx
"""

import shutil
import numpy as np
from torch.utils.data import Dataset
from paths import *
from batchgenerators.utilities.file_and_folder_operations import *
from dataset.loading_dataset import load_dataset
from utilites.split_trainval import load_split, make_split_file
from dataset.augment_data import *
from batchgenerators.augmentations.utils import pad_nd_image
from multiprocessing import Pool


def do_augmentation_train(data_sample, seg_sample, patch_size=None, params=default_3D_augmentation_params,
                          border_val_seg=-1, order_seg=1, order_data=3, ignore_axes=None):
    assert len(data_sample.shape) == 4 and len(seg_sample.shape) == 4, "data, seg shape must be (c, d, h, w)."
    seg_sample = select_channel_from_seg(seg_sample, params["selected_seg_channels"])
    data_sample, seg_sample = augment_spatial_transform(
        data_sample, seg_sample, patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"),
        angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"), angle_z=params.get("rotation_z"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_seg,
        random_crop=params.get("random_crop"),
        p_el_per_sample=params.get("p_eldef"), p_scale_per_sample=params.get("p_scale"),
        p_rot_per_sample=params.get("p_rot")
    )
    data_sample = augment_gaussian_noise(data_sample, p_per_sample=0.1)
    data_sample = augment_gaussian_blur(data_sample, (0.5, 1.), different_sigma_per_channel=True,
                                        p_per_channel=0.5, p_per_sample=0.2)
    data_sample = augment_brightness_multiplicative(data_sample, multiplier_range=(0.75, 1.25),
                                                    per_channel=True, p_per_sample=0.15)
    if params.get("do_additive_brightness"):
        data_sample = augment_brightness_additive(data_sample, params.get("additive_brightness_mu"),
                                                  params.get("additive_brightness_sigma"),
                                                  True, p_per_sample=params.get("additive_brightness_p_per_sample"))
    data_sample = augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True,
                                   per_channel=True, p_per_sample=0.15)
    data_sample = augment_linear_downsampling(data_sample, zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                              channels=None, order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                              ignore_axes=ignore_axes)
    data_sample = augment_gamma(data_sample, params.get("gamma_range"), True, True,
                                retain_stats=params.get("gamma_retain_stats"),
                                p_per_sample=0.1)  # inverted gamma
    if params.get("do_gamma"):
        data_sample = augment_gamma(data_sample, params.get("gamma_range"), False, True,
                                    retain_stats=params.get("gamma_retain_stats"),
                                    p_per_sample=params["p_gamma"])
    if params.get("do_mirror") or params.get("mirror"):
        data_sample, seg_sample = augment_mirroring(data_sample, seg_sample, params.get("mirror_axes"))
    seg_sample = remove_seg_label(seg_sample, -1, 0)
    return data_sample, seg_sample


def do_augmentation_test(data_sample, seg_sample, params=default_3D_augmentation_params):
    seg_sample = remove_seg_label(seg_sample, -1, 0)
    seg_sample = select_channel_from_seg(seg_sample, params["selected_seg_channels"])
    return data_sample, seg_sample


def random_crop_3d(img, seg, patch_size):
    assert len(img.shape)==4
    random_x_max = img.shape[1] - patch_size[0]
    random_y_max = img.shape[2] - patch_size[1]
    random_z_max = img.shape[3] - patch_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None, None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[:, x_random:x_random + patch_size[0], y_random:y_random + patch_size[1],
               z_random:z_random + patch_size[2]]
    crop_label = seg[:, x_random:x_random + patch_size[0], y_random:y_random + patch_size[1],
                 z_random:z_random + patch_size[2]]

    return crop_img, crop_label


def to_one_hot(x, n):
    return np.eye(n)[x.reshape(-1)].reshape(*x.shape, n)


class LungDataset(Dataset):
    def __init__(self, preprocessing_dir, cases_lists, final_patch_size=None, is_training=True,
                 oversample_force_hard_label=4):
        self.preprocessing_dir = preprocessing_dir
        self.final_dataset_dir = join(self.preprocessing_dir, preprocessed_data_identifer)
        self.dataset_info = load_dataset(self.final_dataset_dir, cases_lists)
        self.cases_lists = list(cases_lists)
        self.final_patch_size = final_patch_size
        self.is_training = is_training
        self.data_aug_params = None
        self.set_augment_parameters()
        if is_training:
            if self.final_patch_size is not None:
                self.basic_patch_size = get_patch_size(final_patch_size,
                                                       self.data_aug_params['rotation_x'],
                                                       self.data_aug_params['rotation_y'],
                                                       self.data_aug_params['rotation_z'],
                                                       self.data_aug_params['scale_range'])
            else:
                self.basic_patch_size = None
        else:
            self.basic_patch_size = final_patch_size

        # self.cupy_speed = cupy_speed  # developing
        self.oversample_foreground_percent = 0.5
        self.oversample_force_hard_label = oversample_force_hard_label
        self.oversample_force_hard_label_percent = 0.3

    def set_augment_parameters(self):
        self.data_aug_params = default_3D_augmentation_params
        self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.final_patch_size

    def __len__(self):
        return len(self.cases_lists)

    def __getitem__(self, item):
        data_sample, seg_sample = self.get_training_data(self.cases_lists[item])
        if self.is_training:
            data_sample, seg_sample = do_augmentation_train(data_sample, seg_sample, patch_size=self.final_patch_size,
                                                            params=self.data_aug_params)
        else:
            data_sample, seg_sample = do_augmentation_test(data_sample, seg_sample, self.data_aug_params)
            if self.final_patch_size is not None and any([i<j for i, j in zip(data_sample.shape[1:], self.final_patch_size)]):
                data_sample = pad_nd_image(data_sample, self.final_patch_size, mode='constant', **{'constant_values': 0})
                seg_sample = pad_nd_image(seg_sample, self.final_patch_size, mode='constant', **{'constant_values': 0})
        return self.cases_lists[item], data_sample, seg_sample[0]  # data (c, d, h, w), seg (d, h, w)

    def get_training_data(self, case_identifier):
        if isfile(self.dataset_info[case_identifier]['data_file'][:-4]+".npy"):
            case_all_data = np.load(self.dataset_info[case_identifier]['data_file'][:-4] + ".npy")
        else:
            case_all_data = np.load(self.dataset_info[case_identifier]['data_file'])['data']
        assert len(case_all_data.shape) == 4, "case_all_data length of shape must be 4."
        if self.basic_patch_size is None and self.final_patch_size is None:
            return case_all_data[:-1], case_all_data[-1:]
        # if all([i<j for i, j in zip(case_all_data.shape[1:], self.patch_size)]):
        #     case_all_data = pad_nd_image(case_all_data, self.patch_size)
        if np.random.uniform() < self.oversample_foreground_percent:
            force_fg = True
        else:
            force_fg = False
        if np.random.uniform() < self.oversample_force_hard_label_percent:
            force_hard_label = True
        else:
            force_hard_label = False

        need_to_pad = (np.array(self.basic_patch_size) - np.array(self.final_patch_size)).astype(int)
        for d in range(3):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + case_all_data.shape[d + 1] < self.basic_patch_size[d]:
                need_to_pad[d] = self.basic_patch_size[d] - case_all_data.shape[d + 1]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        shape = case_all_data.shape[1:]
        lb_x = - need_to_pad[0] // 2
        ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.basic_patch_size[0]
        lb_y = - need_to_pad[1] // 2
        ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.basic_patch_size[1]
        lb_z = - need_to_pad[2] // 2
        ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.basic_patch_size[2]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg:
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
        else:
            # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
            foreground_classes = np.array(self.dataset_info[case_identifier]['properties']['classes'])
            foreground_classes = foreground_classes[foreground_classes > 0]
            if len(foreground_classes) == 0:
                selected_class = 0
            else:
                if force_hard_label:
                    selected_class = self.oversample_force_hard_label
                else:
                    selected_class = np.random.choice(foreground_classes)
            # print(force_fg, force_hard_label, selected_class)
            voxels_of_that_class = np.argwhere(case_all_data[-1] == selected_class)

            if len(voxels_of_that_class) != 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]

                bbox_x_lb = max(lb_x, selected_voxel[0] - self.basic_patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.basic_patch_size[1] // 2)
                bbox_z_lb = max(lb_z, selected_voxel[2] - self.basic_patch_size[2] // 2)
            else:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

        bbox_x_ub = bbox_x_lb + self.basic_patch_size[0]
        bbox_y_ub = bbox_y_lb + self.basic_patch_size[1]
        bbox_z_ub = bbox_z_lb + self.basic_patch_size[2]

        valid_bbox_x_lb = max(0, bbox_x_lb)
        valid_bbox_x_ub = min(shape[0], bbox_x_ub)
        valid_bbox_y_lb = max(0, bbox_y_lb)
        valid_bbox_y_ub = min(shape[1], bbox_y_ub)
        valid_bbox_z_lb = max(0, bbox_z_lb)
        valid_bbox_z_ub = min(shape[2], bbox_z_ub)

        case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                      valid_bbox_y_lb:valid_bbox_y_ub,
                                      valid_bbox_z_lb:valid_bbox_z_ub]

        case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                          (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                          (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                          (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                     mode='constant')

        case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                            (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                       mode='constant', **{'constant_values': -1})

        return case_all_data_donly, case_all_data_segonly


class LungDatasetOffline(Dataset):
    def __init__(self, preprocessing_dir, fold=1, actual_patch_size=(128, 192, 192), is_training=True):
        self.net_inputs_fold_dir = join(preprocessing_dir, preprocessed_net_inputs, "fold_"+str(fold))
        self.info_json = load_json(join(self.net_inputs_fold_dir, "info.json"))
        self.patch_size = self.info_json['patch_size']
        self.fold = self.info_json['fold']
        self.max_aug_per_case = self.info_json['max_aug_per_case']
        self.actual_patch_size = actual_patch_size
        self.prefix = "train" if is_training else "val"
        if is_training:
            self.files_list = subfiles(self.net_inputs_fold_dir, True, prefix=self.prefix)
        else:
            self.files_list = subfiles(self.net_inputs_fold_dir, True, prefix=self.prefix)
        cases_lists = [os.path.basename(f)[:-4] for f in self.files_list]
        self.cases_lists = list(set([i.split("_")[1] for i in cases_lists]))
        self.dataset_info = load_dataset(join(preprocessing_dir, preprocessed_data_identifer), self.cases_lists)
        self.global_min_points = None

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, item):
        all_data = np.load(self.files_list[item])

        if any(np.array(self.actual_patch_size) < np.array(self.patch_size)):
            if self.global_min_points is not None:
                min_points = self.global_min_points
            else:
                min_points = [size//2-self.actual_patch_size[i]//2 for i, size in enumerate(self.patch_size)]
                self.global_min_points = min_points
            # print(min_points, self.actual_patch_size)
            all_data = all_data[:, min_points[0]:min_points[0] + self.actual_patch_size[0],
                                min_points[1]:min_points[1] + self.actual_patch_size[1],
                                min_points[2]:min_points[2] + self.actual_patch_size[2]]

        data = all_data[:-1]
        seg = all_data[-1]
        return os.path.basename(self.files_list[item])[:-4].split("_")[1], data, seg


class Generate_net_inputs(object):
    def __init__(self, preprocessed_base_dir, fold=1, patch_size=(128, 192, 192), max_aug_per_case=10,
                 overwrite=False, processor=8):
        self.processor = processor
        net_inputs_fold_dir = join(preprocessed_base_dir, preprocessed_net_inputs, "fold_"+str(fold))
        maybe_mkdir_p(net_inputs_fold_dir)
        self.net_inputs_fold_dir = net_inputs_fold_dir
        if overwrite or isfile(net_inputs_fold_dir):
            shutil.rmtree(net_inputs_fold_dir)
        maybe_mkdir_p(net_inputs_fold_dir)
        self.train_cases, self.val_cases = load_split(fold)
        self.train_dataset = LungDataset(preprocessed_output_dir, self.train_cases, patch_size, True)
        self.val_dataset = LungDataset(preprocessed_output_dir, self.val_cases, patch_size, False)
        self.max_aug_per_case = max_aug_per_case

    def aug_generate(self):
        p = Pool(self.processor)
        print("***** train cases *****")
        for i in range(len(self.train_dataset)):
            print("##### {} #####".format(self.train_cases[i]))
            p.map(self._generate_case_npy, zip([i]*self.max_aug_per_case,
                                               [self.net_inputs_fold_dir]*self.max_aug_per_case,
                                               ["%04d"%i for i in range(self.max_aug_per_case)],
                                               ["train"]*self.max_aug_per_case))
        p.close()
        p.join()
        p2 = Pool(self.processor)
        print("***** validation cases *****")
        for i in range(len(self.val_dataset)):
            print("##### {} #####".format(self.val_cases[i]))
            p2.map(self._generate_case_npy, zip([i] * self.max_aug_per_case,
                                                    [self.net_inputs_fold_dir] * self.max_aug_per_case,
                                                    ["%03d" % i for i in range(self.max_aug_per_case)],
                                                    ["val"] * self.max_aug_per_case))
        p2.close()
        p2.join()

    def _generate_case_npy(self, args):
        idx, out_path, cnt, trainval_str = args
        if not isinstance(cnt, str):
            cnt = str(cnt)
        if trainval_str == "train":
            case_id, data, seg = self.train_dataset[idx] # data (c, d, h, w)   seg (d, h, w)
        else:
            case_id, data, seg = self.val_dataset[idx]
        seg = seg[None]
        all_data = np.vstack([data, seg])
        np.save(join(out_path, trainval_str+"_"+case_id+"_"+cnt+".npy"), all_data)


if __name__ == '__main__':
    # from matplotlib import pyplot as plt
    # if not os.path.isfile(join(preprocessed_output_dir, "trainval_splits.pkl")):
    #     make_split_file(preprocessed_output_dir)
    # train_keys, val_keys = load_split(1)
    # train_dataset = LungDataset(preprocessed_output_dir, train_keys, (128, 196, 196), is_training=True)
    # slice_idx = 50
    # i = 10  # np.random.choice(len(train_dataset), 1)
    # case_id, data, seg = train_dataset[i]
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # axes[0].imshow(data[0][slice_idx], plt.cm.gray, interpolation='nearest')
    # axes[0].axis('off')
    # axes[1].imshow(seg[slice_idx], plt.cm.gray, interpolation='nearest')
    # axes[1].axis('off')
    # plt.tight_layout()
    # plt.show()
    # print("after transform shape:{}, {}, seg unique:{}".format(data.shape, seg.shape, np.unique(seg)))

    from collections import OrderedDict
    patch_size = (128, 192, 192)
    fold = 1
    max_aug_per_case = 10
    generator = Generate_net_inputs(preprocessed_output_dir, fold, patch_size, max_aug_per_case)
    generator.aug_generate()

    generator_info_json = OrderedDict()
    generator_info_json["patch_size"] = patch_size
    generator_info_json["fold"] = fold
    generator_info_json["max_aug_per_case"] = max_aug_per_case
    save_json(generator_info_json, join(preprocessed_output_dir, preprocessed_net_inputs, "fold_"+str(fold), "info.json"))
