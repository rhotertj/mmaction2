# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from mmengine.dataset import BaseDataset
from ..utils import get_root_logger
from .builder import DATASETS


@DATASETS.register_module()
class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. Only applicable to UCF or
            HMDB. Allowed choiced are 'train1', 'test1', 'train2', 'test2',
            'train3', 'test3'. Default: None.
        class_prob (dict | None): The per class sampling probability. If not
            None, it will override the class_prob calculated in
            BaseDataset.__init__(). Default: None.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 class_prob=None):
        modality = 'Pose'
        # split, applicable to ucf or hmdb
        self.split = split

        super().__init__(ann_file, pipeline=pipeline)

        if class_prob is not None:
            self.class_prob = class_prob

        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

    def load_data_list(self, ann_file):
        """Load annotation file to get video information."""
        assert ann_file.endswith('.pkl')
        return self.load_pkl_annotations(ann_file)

    def load_pkl_annotations(self, ann_file):
        data = mmcv.load(ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            data = [x for x in data if x[identifier] in split[self.split]]
            
        return data
