# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.data import BaseDataElement


class SkeDataSample(BaseDataElement):

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self.set_field(value, '_label')

    @label.deleter
    def label(self):
        del self._label

    @property
    def frame_dir(self):
        return self._frame_dir

    @frame_dir.setter
    def frame_dir(self, value):
        self.set_field(value, '_frame_dir')

    @frame_dir.deleter
    def frame_dir(self):
        del self._frame_dir
