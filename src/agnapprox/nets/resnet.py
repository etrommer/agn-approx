"""
Class definition for ResNet Approximate NN
"""
import logging
from typing import Optional

import torch.optim as optim

from .approxnet import ApproxNet
from .base import resnet

logger = logging.getLogger(__name__)

# pylint: disable=too-many-ancestors
class ResNet(ApproxNet):
    """
    Definition of training hyperparameters for
    approximate ResNet
    """

    def __init__(self, resnet_size: Optional[str] = "ResNet8", **kwargs):
        super().__init__(**kwargs)

        self.name = resnet_size
        if self.name == "ResNet8":
            self.model = resnet.resnet8()
        if self.name == "ResNet14":
            self.model = resnet.resnet14()
        if self.name == "ResNet20":
            self.model = resnet.resnet20()
        if self.name == "ResNet32":
            self.model = resnet.resnet32()

        self.topk: tuple = (1,)
        self.epochs = {
            "baseline": 180,
            "noise": 4,
            "qat": 30,
            "approx": 4,
        }
        self.num_gpus: int = 1
        self.gather_noisy_modules()

    def _baseline_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[90, 130, 160], gamma=0.1
        )
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]

    def _gs_optimizers(self):
        pass
