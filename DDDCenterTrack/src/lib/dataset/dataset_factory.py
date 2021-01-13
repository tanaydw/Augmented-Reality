from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os

from .datasets.kitti import KITTI
from .datasets.nuscenes import nuScenes
from .datasets.custom_dataset import CustomDataset

dataset_factory = {
    'custom': CustomDataset,
    'kitti': KITTI,
    'nuscenes': nuScenes,
}


def get_dataset(dataset):
    return dataset_factory[dataset]
