# -*- coding: utf-8 -*-
"""
# @Time    : May/21/2020
# @Author  : zhx
"""
import numpy as np
from utilites.utils import get_patient_identifiers_from_cropped_files
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict
from skimage.morphology import label
from paths import *


def load_properties_of_cropped(folder_with_cropped_data, case_identifier):
    with open(join(folder_with_cropped_data, "%s.pkl" % case_identifier), 'rb') as f:
        properties = pickle.load(f)
    return properties


def get_sizes_and_spacings_after_cropping(folder_with_cropped_data, to_dict=False):
    case_identifiers = get_patient_identifiers_from_cropped_files(folder_with_cropped_data)
    sizes = []
    spacings = []
    for c in case_identifiers:
        properties = load_properties_of_cropped(folder_with_cropped_data, c)
        sizes.append(properties["size_after_cropping"])
        spacings.append(properties["original_spacing"])

    if not to_dict:
        return sizes, spacings
    else:
        return {c:j for c, j in zip(case_identifiers, sizes)}, {c:j for c, j in zip(case_identifiers, spacings)}

class DatasetAnalyzer(object):
    def __init__(self, folder_with_cropped_data, overwrite=True, num_processes=8):
        """
        :param folder_with_cropped_data:
        :param overwrite: If True then precomputed values will not be used and instead recomputed from the data.
        False will allow loading of precomputed values. This may be dangerous though if some of the code of this class
        was changed, therefore the default is True.
        """
        self.num_processes = num_processes
        self.overwrite = overwrite
        self.folder_with_cropped_data = folder_with_cropped_data
        self.sizes = self.spacings = None
        self.patient_identifiers = get_patient_identifiers_from_cropped_files(self.folder_with_cropped_data)
        assert isfile(join(self.folder_with_cropped_data, "dataset.json")), \
            "dataset.json needs to be in folder_with_cropped_data"

        self.props_per_case_file = join(self.folder_with_cropped_data, "props_per_case.pkl")
        self.intensityproperties_file = join(self.folder_with_cropped_data, "intensityproperties.pkl")
        self.dataset_properties_file = join(self.folder_with_cropped_data, "dataset_properties.pkl")

    def analyze_dataset(self, collect_intensityproperties=True):
        # get all spacings and sizes of dict format (case_identifiers)
        sizes, spacings = get_sizes_and_spacings_after_cropping(self.folder_with_cropped_data)

        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        class_dct, segmentation_props_per_patient = self.analyse_segmentations()
        all_classes = np.array([int(i) for i in class_dct.keys()])
        all_classes = all_classes[all_classes > 0]

        # modalities
        modalities = self.get_modalities()

        # collect intensity information
        if collect_intensityproperties:
            intensityproperties = self.collect_intensity_properties(len(modalities))
        else:
            intensityproperties = None

        # size reduction by cropping
        size_reductions = self.get_size_reduction_by_cropping()

        self.dataset_properties = dict()
        self.dataset_properties['all_sizes'] = sizes
        self.dataset_properties['all_spacings'] = spacings
        self.dataset_properties['segmentation_props_per_patient'] = segmentation_props_per_patient
        self.dataset_properties['class_dct'] = class_dct  # {int: class name}
        self.dataset_properties['all_classes'] = all_classes
        self.dataset_properties['modalities'] = modalities  # {idx: modality name}
        self.dataset_properties['intensityproperties'] = intensityproperties
        self.dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        save_pickle(self.dataset_properties, self.dataset_properties_file)
        return self.dataset_properties



    def get_classes(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        return datasetjson['labels']

    def analyse_segmentations(self):
        class_dct = self.get_classes()
        all_classes = np.array([int(i) for i in class_dct.keys()])
        all_classes = all_classes[all_classes > 0]  # remove background

        if self.overwrite or not isfile(self.props_per_case_file):
            p = Pool(self.num_processes)
            res = p.map(self._load_seg_analyze_classes, zip(self.patient_identifiers,
                                                            [all_classes] * len(self.patient_identifiers)))
            p.close()
            p.join()

            props_per_patient = OrderedDict()
            for p, (unique_classes, all_in_one_region, voxels_per_class, region_volume_per_class) in \
                    zip(self.patient_identifiers, res):
                props = dict()
                props['has_classes'] = unique_classes
                props['only_one_region'] = all_in_one_region
                props['volume_per_class'] = voxels_per_class
                props['region_volume_per_class'] = region_volume_per_class
                props_per_patient[p] = props

            save_pickle(props_per_patient, self.props_per_case_file)
        else:
            props_per_patient = load_pickle(self.props_per_case_file)
        return class_dct, props_per_patient

    def _load_seg_analyze_classes(self, args):
        """
        1) what class is in this training case?
        2) what is the size distribution for each class?
        3) what is the region size of each class?
        4) check if all in one region
        :return:
        """
        patient_identifier, all_classes = args
        seg = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data'][-1]
        pkl = load_pickle(join(self.folder_with_cropped_data, patient_identifier) + ".pkl")
        vol_per_voxel = np.prod(pkl['itk_spacing'])

        # ad 1)
        unique_classes = np.unique(seg)

        # 4) check if all in one region
        regions = list()
        regions.append(list(all_classes))
        for c in all_classes:
            regions.append((c, ))

        all_in_one_region = self._check_if_all_in_one_region((seg, regions))

        # 2 & 3) region sizes
        volume_per_class, region_sizes = self._collect_class_and_region_sizes((seg, all_classes, vol_per_voxel))

        return unique_classes, all_in_one_region, volume_per_class, region_sizes

    @staticmethod
    def _check_if_all_in_one_region(args):
        seg, regions = args
        res = OrderedDict()
        for r in regions:
            new_seg = np.zeros(seg.shape)
            for c in r:
                new_seg[seg == c] = 1
            labelmap, numlabels = label(new_seg, return_num=True)
            if numlabels != 1:
                res[tuple(r)] = False
            else:
                res[tuple(r)] = True
        return res

    @staticmethod
    def _collect_class_and_region_sizes(args):
        seg, all_classes, vol_per_voxel = args
        volume_per_class = OrderedDict()
        region_volume_per_class = OrderedDict()
        for c in all_classes:
            region_volume_per_class[c] = []
            volume_per_class[c] = np.sum(seg == c) * vol_per_voxel
            labelmap, numregions = label(seg == c, return_num=True)
            for l in range(1, numregions + 1):
                region_volume_per_class[c].append(np.sum(labelmap == l) * vol_per_voxel)
        return volume_per_class, region_volume_per_class

    def get_modalities(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        modalities = datasetjson["modality"]
        modalities = {int(k): modalities[k] for k in modalities.keys()}
        return modalities

    def collect_intensity_properties(self, num_modalities):
        if self.overwrite or not isfile(self.intensityproperties_file):
            p = Pool(self.num_processes)

            results = OrderedDict()
            for mod_id in range(num_modalities):
                results[mod_id] = OrderedDict()
                v = p.map(self._get_voxels_in_foreground, zip(self.patient_identifiers,
                                                              [mod_id] * len(self.patient_identifiers)))

                w = []
                for iv in v:
                    w += iv

                median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._compute_stats(w)

                local_props = p.map(self._compute_stats, v)
                props_per_case = OrderedDict()
                for i, pat in enumerate(self.patient_identifiers):
                    props_per_case[pat] = OrderedDict()
                    props_per_case[pat]['median'] = local_props[i][0]
                    props_per_case[pat]['mean'] = local_props[i][1]
                    props_per_case[pat]['sd'] = local_props[i][2]
                    props_per_case[pat]['mn'] = local_props[i][3]
                    props_per_case[pat]['mx'] = local_props[i][4]
                    props_per_case[pat]['percentile_99_5'] = local_props[i][5]
                    props_per_case[pat]['percentile_00_5'] = local_props[i][6]

                results[mod_id]['local_props'] = props_per_case
                results[mod_id]['median'] = median
                results[mod_id]['mean'] = mean
                results[mod_id]['sd'] = sd
                results[mod_id]['mn'] = mn
                results[mod_id]['mx'] = mx
                results[mod_id]['percentile_99_5'] = percentile_99_5
                results[mod_id]['percentile_00_5'] = percentile_00_5

            p.close()
            p.join()
            save_pickle(results, self.intensityproperties_file)
        else:
            results = load_pickle(self.intensityproperties_file)
        return results

    def _get_voxels_in_foreground(self, args):
        patient_identifier, modality_id = args
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data']
        modality = all_data[modality_id]
        mask = all_data[-1] > 0
        voxels = list(modality[mask][::10]) # no need to take every voxel
        return voxels

    @staticmethod
    def _compute_stats(voxels):
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

    def get_size_reduction_by_cropping(self):
        size_reduction = OrderedDict()
        for p in self.patient_identifiers:
            props = load_properties_of_cropped(self.folder_with_cropped_data, p)
            shape_before_crop = props["original_size_of_raw_data"]
            shape_after_crop = props['size_after_cropping']
            size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
            size_reduction[p] = size_red
        return size_reduction
