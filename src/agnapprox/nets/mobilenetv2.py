"""
Class definition for MobileNetV2 Approximate NN
"""
import logging

import torch
import torchvision

from .approxnet import ApproxNet

logger = logging.getLogger(__name__)


class MobileNetV2(ApproxNet):
    """
    Definition of training hyperparameters for
    approximate MobileNetV2
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()

        self.name = "MobileNetV2"
        self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)

        # Replace last layer with randomly initialized layer of correct size
        if num_classes != 1000:
            self.model.classifier[1] = torch.nn.Linear(1280, num_classes)

        self.topk = (1, 5)
        self.epochs: dict = {
            "baseline": 30,
            "qat": 8,
            "gradient_search": 5,
            "approx": 2,
        }
        self.num_gpus = 1
        self.gather_noisy_modules()

    def _baseline_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=3e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return [optimizer], [scheduler]

    def _gs_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]
