# -*- coding: utf-8 -*-
"""
# @Time    : May/18/2020
# @Author  : zhx
"""

import numpy as np
import SimpleITK as sitk
from paths import *
from batchgenerators.utilities.file_and_folder_operations import *

class GalaTrainer(object):
    def __init__(self, preprocessed):