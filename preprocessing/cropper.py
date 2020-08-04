# -*- coding: utf-8 -*-
"""
# @Time    : May/19/2020
# @Author  : zhx
"""

import SimpleITK as sitk
import numpy as np
import shutil
from skimage import measure
from multiprocessing import Pool
from paths import raw_splited_dir, raw_cropped_data_dir
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from matplotlib import pyplot as plt


def get_case_identifier(case):
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0]
    return case_identifier


def lung_mask(data_itk, seg_itk=None, case_identifier=None, test_mask_contain_seg=False): # numpy.ndarray  [depth,height,width]
    # 1 阈值分割
    # 2 种子填充
    # 3 图像形态学
    # 4 保留最大连通域
    ################################################3
    # 获取体数据的尺寸
    size = data_itk.GetSize()  # width, height, depth
    # 获取体数据的空间尺寸
    spacing = data_itk.GetSpacing()
    # 将体数据转为numpy数组
    data_npy = sitk.GetArrayFromImage(data_itk)

    # 根据CT值，将数据二值化
    data_npy[data_npy >= -700] = 1
    data_npy[data_npy < - 700] = 0

    # 生成阈值图像
    threshold = sitk.GetImageFromArray(data_npy)
    threshold.SetSpacing(spacing)
    # 去除背板的二值
    bm = sitk.BinaryMorphologicalOpeningImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(3)
    bm.SetForegroundValue(1)
    threshold = bm.Execute(threshold)

    # 利用种子生成算法，填充空气
    ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
    ConnectedThresholdImageFilter.SetLower(0)
    ConnectedThresholdImageFilter.SetUpper(0)
    ConnectedThresholdImageFilter.SetSeedList([(0, 0, 0), (size[0]-1, size[1]-1, 0)])

    # 得到body的mask，此时body部分是0，所以反转一下
    bodymask = ConnectedThresholdImageFilter.Execute(threshold)
    bodymask = sitk.ShiftScale(bodymask, -1, -1) #像素反转

    # 用bodymask减去threshold，得到初步的lung的mask
    temp = sitk.GetImageFromArray(sitk.GetArrayFromImage(bodymask) - sitk.GetArrayFromImage(threshold))
    temp.SetSpacing(spacing)
    # 利用形态学来去掉一定的肺部的小区域
    bm = sitk.BinaryMorphologicalClosingImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(3)
    bm.SetForegroundValue(1)
    temp = bm.Execute(temp)

    # bm = sitk.BinaryDilateImageFilter()
    # bm.SetKernelType(sitk.sitkBall)
    # bm.SetKernelRadius(5)
    # bm.SetForegroundValue(1)
    # lungmask = bm.Execute(temp)

    # 利用measure来计算连通域
    lungmaskarray = sitk.GetArrayFromImage(temp)
    labels = measure.label(lungmaskarray, connectivity=2)
    props = measure.regionprops(labels)

    # 计算每个连通域的体素的个数
    numPix = [p.area for p in props]
    print("3D regions pixels:")
    print(sorted(numPix, reverse=True)[:10])

    mask_array = np.zeros_like(labels, dtype=np.float32)
    # 最大连通域的体素个数，也就是肺部, 最大连通区域置为1
    maxnum = max(numPix)
    mask_array[labels == numPix.index(maxnum) + 1] = 1

    bbox_mask = get_bbox_from_mask(mask_array, 0, expand_slices=(2,2,2))
    bbox_seg = get_bbox_from_mask(sitk.GetArrayFromImage(seg_itk))[:, ::-1, :] # reverse height, this is bug for me.
    if test_mask_contain_seg:
        oricount = 0
        flag = 0
        if case_identifier is None: raise ValueError("case_identifier is need if test_mask_contain_seg is True")
        # mindepth_mask, maxdepth_mask, minheight_mask, maxheight_mask, minwidth_mask, maxwidth_mask = bbox_mask
        # mindepth_seg, maxdepth_seg, minheight_seg, maxheight_seg, minwidth_seg, maxwidth_seg = bbox_seg
        ori = ["depth", "height", "width"]
        minormax = ["min", "max"]
        print("Test lung mask:")
        print(bbox_mask, "\n", bbox_seg)
        for i in range(3):
            if bbox_mask[i][0] > bbox_seg[i][0]:
                oricount += 1
                flag = 1
                print("#####!!!the {0} {1} of lung mask > lung seg".format(ori[i], minormax[0]))
            if bbox_mask[i][1] < bbox_seg[i][1]:
                oricount += 1
                flag = 1
                print("#####!!!the {0}  {1} of lung mask < lung seg".format(ori[i], minormax[1]))
        print("Done.")
        return mask_array, bbox_mask, bbox_seg, oricount, flag
    # l = sitk.GetImageFromArray(label)
    # l.CopyInformation(data_itk)
    return bbox_mask, bbox_seg


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def lung_crop(data_npy, bbox, seg_npy=None):
    cropped_data = []
    for c in range(data_npy.shape[0]):
        cropped = crop_to_bbox(data_npy[c], bbox)
        cropped_data.append(cropped[None])
    data_npy = np.vstack(cropped_data)

    if seg_npy is not None:
        cropped_seg = []
        for c in range(seg_npy.shape[0]):
            cropped = crop_to_bbox(seg_npy[c], bbox)
            cropped_seg.append(cropped[None])
        seg_npy = np.vstack(cropped_seg)

    return data_npy, seg_npy


def get_bbox_from_mask(mask, outside_value=0, expand_slices=(0, 0, 0)): # numpy.ndarray
    mask_voxel_coords = np.where(mask != outside_value)
    mindepthidx = int(np.max([min(mask_voxel_coords[0])-expand_slices[0], 0]))
    maxdepthidx = int(np.min([max(mask_voxel_coords[0])+expand_slices[0], mask.shape[0]-1])) + 1
    minheightidx = int(np.max([min(mask_voxel_coords[1])-expand_slices[1], 0]))
    maxheightidx = int(np.min([max(mask_voxel_coords[1])+expand_slices[1], mask.shape[1]-1])) + 1
    minwidthidx = int(np.max([min(mask_voxel_coords[2])-expand_slices[2], 0]))
    maxwidthidx = int(np.min([max(mask_voxel_coords[2])+expand_slices[2], mask.shape[2]-1])) + 1
    return [[mindepthidx, maxdepthidx], [minheightidx, maxheightidx], [minwidthidx, maxwidthidx]]


class ImageCropper:
    def __init__(self, num_threads, cropped_dir=None):
        self.splited_dir = raw_splited_dir
        self.cropped_dir = cropped_dir
        if self.cropped_dir is not None:
            maybe_mkdir_p(self.cropped_dir)
        self.num_threads = num_threads

    def _cropper(self, args):
        return self.cropper(*args)

    def cropper(self, case, case_identifier, overwrite_existing):
        try:
            print(case_identifier)
            if overwrite_existing \
                    or (not isfile(join(self.cropped_dir, "%s.npz" % case_identifier))
                        or not isfile(join(self.cropped_dir, "%s.pkl" % case_identifier))):
                data, seg, properties = self.lung_crop_for_one_case(case[0], case[1], case_identifier) # one datafile segfile

                all_data = np.vstack((data, seg))
                np.savez_compressed(join(self.cropped_dir, "%s.npz" % case_identifier), data=all_data)
                with open(os.path.join(self.cropped_dir, "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            print("Except for", case_identifier, ", error log below:")
            print(e)
            raise e

    def load_properties(self, case_identifier):
        with open(os.path.join(self.cropped_dir, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.cropped_dir, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

    def lung_crop_for_one_case(self, data_file, seg_file, case_identifier):
        data_itk, data_npy, seg_itk, seg_npy, properties = self.load_case_from_files(data_file, seg_file) # data_npy and seg_npy 4D
        shape_before = data_npy[0].shape
        bbox_mask, bbox_seg = lung_mask(data_itk, seg_itk, case_identifier)
        data_cropped, seg_cropped = lung_crop(data_npy, bbox_mask, seg_npy)  # c,d,h,w
        shape_after = data_cropped[0].shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        properties["lung_crop_bbox"] = bbox_mask
        properties["seg_crop_bbox"] = bbox_seg
        properties['classes'] = np.unique(seg_cropped)
        # seg_cropped[seg_cropped < 1] = 0
        properties["size_after_cropping"] = data_cropped[0].shape
        return data_cropped, seg_cropped, properties

    def load_case_from_files(self, data_file, seg_file):
        assert isinstance(data_file, str) and isinstance(seg_file, str), "case data or seg path must be a str."
        properties = OrderedDict()
        data_itk = sitk.ReadImage(data_file)
        properties["original_size_of_raw_data"] = np.array(data_itk.GetSize())[[2, 1, 0]] # got (Width, Height, Depth)
        properties["original_spacing"] = np.array(data_itk.GetSpacing())[[2, 1, 0]]
        properties["list_of_data_files"] = data_file
        properties["seg_file"] = seg_file
        properties["itk_origin"] = data_itk.GetOrigin()
        properties["itk_spacing"] = data_itk.GetSpacing()
        properties["itk_direction"] = data_itk.GetDirection()
        data_npy = sitk.GetArrayFromImage(data_itk)[None]
        seg_itk = None
        if seg_file is not None:
            seg_itk = sitk.ReadImage(seg_file)
            seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)[:, :, ::-1, :]
        else:
            seg_npy = None
        return data_itk, data_npy.astype(np.float32), seg_itk, seg_npy, properties

    def run_cropping(self, case_lists_files, overwrite_existing=False):
        gt_cropped_dir = join(self.cropped_dir, 'gt_segmentations')
        maybe_mkdir_p(gt_cropped_dir)
        for j, case in enumerate(case_lists_files): # copy train label to cropped_dir/gt_segmentations
            if case[-1] is not None:
                shutil.copy(case[-1], gt_cropped_dir)

        list_of_args = []
        for j, case in enumerate(case_lists_files):
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        if self.num_threads is not None:
            p = Pool(self.num_threads)
        else: p = Pool()
        p.map(self._cropper, list_of_args)
        p.close()
        p.join()


if __name__ == '__main__':
    # test crop bbox contain lung fully.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-om', '--overwrite_mask_error', action='store_true', default=False,
                        help='whether to overwrite mask error dir.')
    args = parser.parse_args()
    raw_splited_dir = "/data1/mzs/zhx/lung_lobe_seg/galaNet_raw_data/splited_data/"
    images_list_files = subfiles(join(raw_splited_dir, "train_images"), join=True, sort=True)
    labels_list_files = subfiles(join(raw_splited_dir, "train_labels"), join=True, sort=True)
    test_images_list_files = subfiles(join(raw_splited_dir, "test_images"), join=True, sort=True)
    test_labels_list_files = subfiles(join(raw_splited_dir, "test_labels"), join=True, sort=True)
    images_list_files.extend(test_images_list_files)
    labels_list_files.extend(test_labels_list_files)
    for img, label in zip(images_list_files, labels_list_files):
        assert os.path.basename(img).split(".nii.gz")[0]==os.path.basename(label).split(".nii.gz")[0]
    case_error_cnt = 0
    ori_error_cnt = 0
    mask_error_out_dir = join(raw_cropped_data_dir, "mask_error")
    if args.overwrite_mask_error and isdir(mask_error_out_dir):
        shutil.rmtree(mask_error_out_dir)
    maybe_mkdir_p(mask_error_out_dir)
    i = 0
    bbox_dict = OrderedDict()
    bbox_dict["cropped_bbox_list(target, seg)"] = []
    for data_file, seg_file in zip(images_list_files, labels_list_files):
        data_itk = sitk.ReadImage(data_file)
        seg_itk = sitk.ReadImage(seg_file)
        case_id = data_file.split(os.sep)[-1].split(".nii.gz")[0]
        print("{} :".format(i))
        print(case_id)
        mask_array, bbox_mask, bbox_seg, oricnt, flag = lung_mask(data_itk, seg_itk, case_id, True)
        bbox_dict["cropped_bbox_list(target, seg)"].append({str(case_id):[bbox_mask, bbox_seg]})
        if flag:
            mask_itk = sitk.GetImageFromArray(mask_array)
            mask_itk.SetSpacing(data_itk.GetSpacing())
            mask_itk.SetDirection(data_itk.GetDirection())
            mask_itk.SetOrigin(data_itk.GetOrigin())
            sitk.WriteImage(mask_itk, join(mask_error_out_dir, "{}.nii.gz".format(case_id)))
        ori_error_cnt += oricnt
        case_error_cnt += flag
        i += 1
    save_json(bbox_dict, join(raw_cropped_data_dir, "bbox_cropped_seg.json"))
    print("#"*50)
    print("case contain rate:{0}\norientation contain rate:{1}".format(case_error_cnt/len(images_list_files),
                                                                       ori_error_cnt/(6*len(images_list_files))))
