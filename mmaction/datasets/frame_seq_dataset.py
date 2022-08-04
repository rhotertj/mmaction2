from collections import OrderedDict
import os
import copy
import warnings
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from mmcv.utils import print_log
from mmaction.datasets.base import BaseDataset
from mmaction.datasets.builder import DATASETS
from mmaction.core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy)


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
        print("Reading", self.ann_file)
        df = pd.read_csv(self.ann_file)
        self.video_infos = df
        self.video_infos["label"] = self.video_infos["label"].astype(np.int64 )
        return df

    def prepare_test_frames(self, idx):
        frame_dict = self.video_infos.iloc[idx].to_dict()
        frame_dict["idx"] = idx
        return self.pipeline(frame_dict)

    def prepare_train_frames(self, idx):
        #print(f"--Get train frame {idx}--")
        frame_dict = self.video_infos.iloc[idx].to_dict()
        frame_dict["idx"] = idx
        return self.pipeline(frame_dict)


    def evaluate(self,
                results,
                metrics='top_k_accuracy',
                metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                logger=None,
                **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = self.video_infos["label"].tolist()

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                    {}).setdefault(
                                                        'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results,
                                                        gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results

