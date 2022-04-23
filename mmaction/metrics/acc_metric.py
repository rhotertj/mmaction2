import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, load
from mmengine.logging import MMLogger
from mmaction.core import top_k_accuracy, mean_class_accuracy

from .builder import METRICS


@METRICS.register_module()
class AccMetric(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 options=('top1', 'top5', 'mean1'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        for opt in options:
            assert opt in ['top1', 'top5', 'mean1']
        self.options = options

    def process(self, data_batch: Sequence[Tuple[Any, dict]],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        # print(data_batch, predictions, type(data_batch), type(predictions))
        for (_, data_sample), pred in zip(data_batch, predictions):
            # TODO: directly convert to json here?
            result = dict()

            # print(len(data_sample), data_sample, pred)
            result['frame_dir'] = data_sample['frame_dir']
            result['label'] = data_sample['label']
            result['pred'] = pred
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()

        preds = [x['pred'] for x in results]
        labels = [x['label'] for x in results]

        top1, top5 = top_k_accuracy(preds, labels, (1, 5))        
        mean1 = mean_class_accuracy(preds, labels)

        eval_results = OrderedDict(top1=top1, top5=top5, mean1=mean1)
        return eval_results