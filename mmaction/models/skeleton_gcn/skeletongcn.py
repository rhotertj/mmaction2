# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from mmaction.core import stack_batch
from mmcv.runner import BaseModule

from .. import builder
from ..builder import RECOGNIZERS


@RECOGNIZERS.register_module()
class SkeletonGCN(BaseModule, metaclass=ABCMeta):
    """Base class for GCN-based action recognition.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature.
            Default: None.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(self, backbone, cls_head=None, train_cfg=None, test_cfg=None):
        super().__init__()
        # record the source of the backbone
        self.backbone_from = 'mmaction2'
        self.backbone = builder.build_backbone(backbone)
        self.cls_head = builder.build_head(cls_head) if cls_head else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.register_buffer('flag', torch.tensor([1.]), False)
        self.init_weights()

    @property
    def device(self):
        return self.flag.device

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

    def forward_train(self, inputs, data_samples):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        losses = dict()

        x = self.extract_feat(inputs)
        output = self.cls_head(x)
        gt_labels = [x.label for x in data_samples]
        gt_labels = torch.Tensor(gt_labels).to(x.device).type(torch.long)
        gt_labels = gt_labels.squeeze(-1)
        loss = self.cls_head.loss(output, gt_labels)
        losses.update(loss)

        return losses

    def forward_test(self, inputs):
        """Defines the computation performed at every call when evaluation and testing."""
        x = self.extract_feat(inputs)
        assert self.with_cls_head
        output = self.cls_head(x)
        return output.data.cpu().numpy()

    def forward(self, data_batch, return_loss=True):
        """Define the computation performed at every call."""
        inputs, data_samples = self.preprocess_data2(data_batch)
        if return_loss:
            ret = self.forward_train(inputs, data_samples)
            loss, log_vars = self._parse_losses(ret)
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch))
            return outputs
        return self.forward_test(inputs)
        
    def preprocess_data(self, data):
        inputs = [data_['inputs'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        data_samples = [data_sample.to(self.device) for data_sample in data_samples]
        inputs = [input.to(self.device) for input in inputs]
        batch_inputs = stack_batch(inputs)

        return batch_inputs, data_samples

    def preprocess_data2(self, data):
        inputs = [data_[0] for data_ in data]
        data_samples = [data_[1] for data_ in data]

        data_samples = [data_sample.to(self.device) for data_sample in data_samples]
        inputs = [input.to(self.device) for input in inputs]
        batch_inputs = stack_batch(inputs)
        
        return batch_inputs, data_samples

    def extract_feat(self, skeletons):
        x = self.backbone(skeletons)
        return x

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data_batch, optimizer, **kwargs):
        return self(data_batch, return_loss=True)

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        return self(data_batch, return_loss=False)
