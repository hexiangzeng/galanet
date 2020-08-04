# -*- coding: utf-8 -*-
"""
# @Time    : May/19/2020
# @Author  : zhx
"""
import random
import dicom2nifti
import numpy as np
import shutil
from multiprocessing import Pool
import SimpleITK as sitk
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from paths import raw_splited_dir

class SplitDataset:
    def __init__(self, dicom_basedir, splited_dir, overwrite_existing=False):
        self.dicom_basedir = dicom_basedir
        self.splited_dir = splited_dir
        if overwrite_existing and isdir(splited_dir):
            shutil.rmtree(splited_dir)
            maybe_mkdir_p(splited_dir)
        self.overwrite_existing = overwrite_existing
        maybe_mkdir_p(splited_dir)
        maybe_mkdir_p(join(splited_dir, "train_images"))
        maybe_mkdir_p(join(splited_dir, "train_labels"))
        maybe_mkdir_p(join(splited_dir, "test_images"))
        maybe_mkdir_p(join(splited_dir, "test_labels"))

        Badcase = OrderedDict()
        Badcase["collect author"] = "zhx"
        Badcase["Shape_Mismatched"] = []
        Badcase["ZipFile_Error"] = []
        Badcase["SLICE_INCREMENT_INCONSISTENT"] = []
        Badcase["other"] = []
        self.Badcase = Badcase

    def split(self, split_rate=0.9):
        if not self.overwrite_existing: return
        datasetjson = OrderedDict()
        datasetjson['name'] = "Pulmonary Lobe Segmentation"
        datasetjson['description'] = "Pulmonary lobe segmentation"
        datasetjson['tensorImageSize'] = "4D"  # Image W,H,D and a channel
        datasetjson['version'] = "1.0"
        datasetjson['author'] = "hexiangzeng"
        datasetjson['modality'] = {
            "0": "CT",
        }
        datasetjson['labels'] = {
            "0": "background",
            "1": "Left Down",
            "2": "Left Up",
            "3": "Right Down",
            "4": "Right Middle",
            "5": "Right Up"
        }
        self.dicom2niigz() # dicom -> nifti
        pass    # LUNA raw -> nifti
        case_lists = subdirs(raw_splited_dir, False) # not join
        exclude_dir = ["train_images", "train_labels", "test_images", "test_labels"]
        case_lists = [c for c in case_lists if c not in exclude_dir]
        print("Detect valid case:", case_lists)
        total = len(case_lists)
        numtrain = int(total*split_rate)
        datasetjson['total'] = total
        datasetjson['numTraining'] = numtrain
        datasetjson['numTest'] = total-numtrain
        datasetjson['training'] = []
        datasetjson['test_images'] = []
        datasetjson['test_labels'] = []
        for i, c in enumerate(case_lists):
            print(i, c)
            if len(subfiles(join(raw_splited_dir, c)))==0:
                print("case {} extract dicom to nifti got error, the folder is empty.".format(c))
                continue #临时存放文件可能出错导致文件夹内容为空
            if i < numtrain:
                shutil.move(join(raw_splited_dir, c, "imaging.nii.gz"), join(raw_splited_dir, "train_images", c + ".nii.gz"))
                shutil.move(join(raw_splited_dir, c, "segmentation.nii.gz"), join(raw_splited_dir, "train_labels", c + ".nii.gz"))
                datasetjson['training'].append({'image': "./train_images/%s.nii.gz" % c, "label": "./train_labels/%s.nii.gz" % c})
            else:
                shutil.move(join(raw_splited_dir, c, "imaging.nii.gz"), join(raw_splited_dir, "test_images", c + ".nii.gz"))
                shutil.move(join(raw_splited_dir, c, "segmentation.nii.gz"), join(raw_splited_dir, "test_labels", c + ".nii.gz"))
                datasetjson['test_images'].append("./test_images/%s.nii.gz" % c)
                datasetjson['test_labels'].append("./test_labels/%s.nii.gz" % c)
        for c in case_lists:
            shutil.rmtree(join(raw_splited_dir, c))
        save_json(datasetjson, os.path.join(raw_splited_dir, "dataset.json"))

    def dicom2niigz(self):
        # dicom -> nifti data(exclude bad dicom slices)and compassed -> split trainset and testset
        # 17.8GB -> 9.9GB for me

        dirlist = subdirs(self.dicom_basedir, True)# 之后加上LUNA16的50肺分割开源数据
        valid_path_list = []
        for i, casepath in enumerate(dirlist):
            print(i, ":")
            im_3D, volume, seg = self.get_volume_from_case_path(casepath)
            if seg is None:
                if volume is None:
                    self.Badcase["ZipFile_Error"].append(casepath.split(os.sep)[-1])
                else:
                    self.Badcase["Shape_Mismatched"].append(casepath.split(os.sep)[-1])
                continue
            del im_3D, volume, seg
            valid_path_list.append(casepath)
        # valid_path_list = dirlist[:5]  # for test
        print("**********\nThe dicom valid path:{0}\n{1}\n**********".format(len(valid_path_list), str(valid_path_list)))
        p = Pool()
        valid_path_list = sorted(valid_path_list)
        output_path_list = [join(raw_splited_dir, os.path.basename(v)) for v in valid_path_list]
        p.map(self.onedicom2niigz, zip(valid_path_list, output_path_list))
        p.close()
        p.join()
        save_json(self.Badcase, join(self.splited_dir, "Badcase.json"))

    def onedicom2niigz(self, args):
        case_path, output_path = args
        try:
            maybe_mkdir_p(output_path)
            dicominfo = dicom2nifti.dicom_series_to_nifti(
                join(case_path, "slices"), join(output_path, "imaging.nii.gz"),
                reorient_nifti=False)
            # save_pickle(dicominfo, join(case_output, "dicominfo.pkl")) # too big
            im_3D, volume, lobe_mask = self.get_volume_from_case_path(case_path)
            if lobe_mask is None:
                self.Badcase["other"].append(case_path.split(os.sep)[-1])
                print("!!!{} path is Bad.".format(case_path.split(os.sep)[-1]))
                return

            seg = sitk.GetImageFromArray(lobe_mask)
            seg.CopyInformation(im_3D)
            sitk.WriteImage(seg, join(output_path, "segmentation.nii.gz"))

            print("PatientID Done:{0}  volume shape:{1}, segmentation shape:{2}".format(
                case_path.split(os.sep)[-1],
                volume.shape,
                lobe_mask.shape))
            del im_3D, volume, lobe_mask, dicominfo

        except Exception as e:
            print("Error:", e)
            print("SLICE_INCREMENT_INCONSISTENT ERROR:{}".format(case_path.split(os.sep)[-1]))
            self.Badcase["SLICE_INCREMENT_INCONSISTENT"].append(case_path.split(os.sep)[-1])

    def get_volume_from_case_path(self, case_path, scan_name="slices", anno_dir_name="annotation", anno_file="annotation.npz"):
        try:
            scan_path = join(case_path, scan_name)
            series_uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(scan_path)
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(scan_path, series_uids[0])
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(file_names)
            im_3D = series_reader.Execute()
            volume = sitk.GetArrayFromImage(im_3D)
            print('{0}:    scan volume:{1}'.format(case_path, volume.shape))

            anno_path = join(case_path, anno_dir_name)
            annotation = np.load(join(anno_path, anno_file))
            # print('    anno:', annotation.files)
            lobe_mask = np.zeros_like(volume, dtype=volume.dtype)
            for k in range(len(annotation.files)):
                label = k + 1
                mask = annotation[annotation.files[k]]
                if lobe_mask.shape != mask.shape:
                    print("!!!!the pung lobe Segmentation shape {0} do not match the volume shape {1}".format(mask.shape, lobe_mask.shape))
                    lobe_mask = None
                    break
                lobe_mask[mask == 1] = label
            # if lobe_mask is not None:
                # print("    ", np.unique(lobe_mask))
            return im_3D, volume, lobe_mask
        except Exception as e:
            print("Error:", str(e))
            return None, None, None


if __name__ == '__main__':
    # raw_splited_dir = "/data0/mzs/zhx/lung_lobe_seg/galaNet_raw_data/splited_data"
    dicom_dir = "/data0/mzs/zhx/Lobe_SegV2/nnUNet_raw/dicom_data"
    sp = SplitDataset(dicom_dir, raw_splited_dir, False)
    sp.split()
