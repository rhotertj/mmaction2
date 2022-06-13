import os
import shutil
import cv2
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from mmaction.datasets.base import BaseDataset
from mmaction.datasets.builder import DATASETS


HBL_ANNOTATIONS_BASE = Path("/nfs/data/mm4spa/handball_hbl/annotations/ballcentered-actions-taxanomy-its")
HBL_VIDEO_BASE = Path("/nfs/data/mm4spa/handball_hbl/video_30_fps")
HBL_IMAGE_BASE = Path("/home/rhotertj/handball_frames") # devbox4

columns = ["id", "file", "flag", "temporal_coordinates", "spatial_coordinates", "metadata"]

@DATASETS.register_module()
class FrameSequenceDataset(BaseDataset):

    def __init__(self,
        ann_file,
        pipeline,
        data_prefix=None,
        test_mode=False,
        multi_class=False,
        num_classes=None,
        start_index=1,
        modality='RGB',
        sample_by_class=False,
        power=0,
        dynamic_length=False):
        super().__init__(ann_file, pipeline, data_prefix, test_mode, multi_class, num_classes, start_index, modality, sample_by_class, power, dynamic_length)

    def load_annotations(self):
        match_annotation_files = os.listdir(HBL_ANNOTATIONS_BASE)
        match_dfs = []
        for maf in match_annotation_files:
            match_df = pd.read_json(HBL_ANNOTATIONS_BASE / maf , lines=True)
            # BASE / video_file.d / frame_start.jpg
            match_df["file"] = match_df["t_start"].apply(lambda t : f"{HBL_IMAGE_BASE}/{maf.split('-')[0]}.mp4.d/{str(t).rjust(6, '0')}.jpg")
            match_df = match_df.drop(columns=["annotator", "t_end"]) # empty
            match_dfs.append(match_df)
            
        self.video_infos = pd.concat(match_dfs)
        return self.video_infos

    # figure out how to supply multiple images from index -> BxIxCxWxH tensor? 
    def prepare_test_frames(self, idx):
        return self.video_infos.iloc[idx]

    def prepare_train_frames(self, idx):
        #print(f"--Get train frame {idx}--")
        frame_dict = self.video_infos.iloc[idx].to_dict()
        frame_dict["idx"] = idx
        frame_dict["labels"] =  0
        return self.pipeline(frame_dict)