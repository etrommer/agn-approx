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
        super().__init__()

        self.model = lenet5.LeNet5(10)

        self.name = "LeNet5"
        self.topk = (1,)
        self.epochs = {
            "baseline": 0,
            "qat": 4,
            "approx": 5,
        }
        self.num_gpus = 1

    # def on_train_epoch_start(self) -> None:
    #     self.model.apply(torch.ao.quantization.disable_fake_quant)
    #     if self.mode == "qat" and self.current_epoch >= 2:
    #         logger.warning("Enabling Fake Quant")
    #         self.model.apply(torch.ao.quantization.disable_observer)
    #         self.model.apply(torch.ao.quantization.enable_fake_quant)

    def _baseline_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.75)
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]
