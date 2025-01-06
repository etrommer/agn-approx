"""
Class definition for MobileViT
"""

import logging
import math
from typing import Optional

import torch
import torchvision
import pytorch_lightning as pl

from .approxnet import ApproxNet
from .base import mobilevit

logger = logging.getLogger(__name__)


class MobileViT(ApproxNet):
    """
    Definition of training hyperparameters for
    approximate MobileViT
    """

    def __init__(self, num_classes: int = 200, **kwargs):
        super().__init__(**kwargs)

        self.name = "MobileViT"

        # MobileViT XXS, but for num_classes != 1000
        dims = [64, 80, 96]
        channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        self.model = mobilevit.MobileViT(
            (64, 64), dims, channels, num_classes, expansion=2, patch_size=(2, 2)
        )
        self.epochs: dict = {
            "baseline": 30,
            "qat": 12,
            "noise": 5,
            "approx": 4,
        }

        self.topk = (1, 5)
        self.num_gpus = 1

    def _baseline_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=0.0002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-5
        )
        # scheduler = .GradualWarmupScheduler(
        #     self.optimizer, multiplier=1.0, total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler
        # )
        # scheduler =
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        if self.tune_bn:
            params = []
            for p in self.parameters():
                p.requires_grad = False
            norm_types = [
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.LayerNorm,
            ]
            for m in self.modules():
                if any([isinstance(m, norm) for norm in norm_types]) or isinstance(
                    m, torch.nn.Linear
                ):
                    params.append(m.weight)
                    m.weight.requires_grad = True
                    if m.bias is not None:
                        params.append(m.bias)
                        m.bias.requires_grad = True
                elif hasattr(m, "bias") and m.bias is not None:
                    params.append(m.bias)
                    m.bias.requires_grad = True
        else:
            params = [p for p in self.parameters()]
        optimizer = torch.optim.Adam(params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=4, eta_min=1e-5
        )
        return [optimizer], [scheduler]

    def _noise_optimizers(self):
        params = [m.stdev for _, m in self.approx_modules]
        optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]
