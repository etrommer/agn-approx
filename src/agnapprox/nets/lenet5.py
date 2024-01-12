"""
Class definition for LeNet5 Approximate NN
"""
import logging
from typing import Optional

import torch.optim as optim
import torch

from agnapprox.nets.approxnet import ApproxNet
from agnapprox.nets.base import lenet5

logger = logging.getLogger(__name__)


# pylint: disable=too-many-ancestors
class LeNet5(ApproxNet):
    """
    Definition of training hyperparameters for
    approximate LeNet5
    """

    def __init__(self, baseline_model: Optional[torch.nn.Module] = None):
        super().__init__(deterministic=True)

        self.model = lenet5.LeNet5(10)

        self.name = "LeNet5"
        self.topk = (1,)
        self.epochs = {
            "baseline": 10,
            "qat": 6,
            "approx": 6,
            "noise": 6,
        }
        self.num_gpus = 1

    def _baseline_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.75)
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=5e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4, 5])
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=5e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4, 5])
        return [optimizer], [scheduler]

    def _noise_optimizers(self):
        params = [m.stdev for _, m in self.approx_modules]
        optimizer = optim.SGD(params, lr=2e-2)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [3, 4])
        return [optimizer], [scheduler]

    def _prune_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=5e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4, 5])
        return [optimizer], [scheduler]
