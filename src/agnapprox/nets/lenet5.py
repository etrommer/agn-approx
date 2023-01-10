"""
Class definition for LeNet5 Approximate NN
"""
import logging

import torch.optim as optim

from agnapprox.nets.approxnet import ApproxNet
from agnapprox.nets.base import lenet5

logger = logging.getLogger(__name__)

# pylint: disable=too-many-ancestors
class LeNet5(ApproxNet):
    """
    Definition of training hyperparameters for
    approximate LeNet5
    """

    def __init__(self):
        super().__init__()
        self.model = lenet5.LeNet5(10)
        self.name = "LeNet5"
        self.topk = (1,)
        self.epochs = {
            "baseline": 10,
            "qat": 3,
            "noise": 3,
            "approx": 3,
        }
        self.num_gpus = 1
        self.gather_noisy_modules()

    def _baseline_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.75)
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]

    def _gs_optimizers(self):
        pass
