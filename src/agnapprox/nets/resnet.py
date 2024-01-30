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

    def __init__(
        self, resnet_size: Optional[str] = "ResNet8", num_classes: int = 10, **kwargs
    ):
        super().__init__(**kwargs)

        self.name = resnet_size
        if self.name == "ResNet8":
            self.model = resnet.resnet8(num_classes)
        elif self.name == "ResNet14":
            self.model = resnet.resnet14(num_classes)
        elif self.name == "ResNet20":
            self.model = resnet.resnet20(num_classes)
        elif self.name == "ResNet32":
            self.model = resnet.resnet32(num_classes)
        else:
            raise ValueError(f"Unknown ResNet size: {resnet_size}")

        self.topk: tuple = (1,)
        self.epochs = {
            "baseline": 180,
            "noise": 10,
            "qat": 30,
            "approx": 10,
        }
        self.num_gpus: int = 1

    def _baseline_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[90, 135], gamma=0.1
        )
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8])
        return [optimizer], [scheduler]

    def _noise_optimizers(self):
        params = [m.stdev for _, m in self.approx_modules]
        optimizer = optim.SGD(params, lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 10])
        return [optimizer], [scheduler]
