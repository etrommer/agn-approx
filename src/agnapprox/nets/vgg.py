"""
Class definition for VGG Approximate NN
"""
import logging
from typing import Optional

import torch
import torchvision

from .approxnet import ApproxNet

logger = logging.getLogger(__name__)


# pylint: disable=too-many-ancestors
class VGG(ApproxNet):
    """
    Definition of training hyperparameters for
    approximate VGG
    """

    def __init__(
        self,
        vgg_size: Optional[str] = "VGG11",
        num_classes: int = 200,
        pretrained: bool = True,
    ):
        super().__init__()

        self.name = vgg_size
        if self.name == "VGG11":
            self.model = torchvision.models.vgg11_bn(pretrained=pretrained)
        if self.name == "VGG13":
            self.model = torchvision.models.vgg13_bn(pretrained=pretrained)
        if self.name == "VGG16":
            self.model = torchvision.models.vgg16_bn(pretrained=pretrained)
        if self.name == "VGG19":
            self.model = torchvision.models.vgg19_bn(pretrained=pretrained)

        # Replace last layer with randomly initialized layer of correct size
        if num_classes != 1000:
            self.model.classifier[6] = torch.nn.Linear(4096, num_classes)

        self.topk = (1, 5)
        self.epochs: dict = {
            "baseline": 30,
            "qat": 8,
            "gradient_search": 3,
            "approx": 2,
        }
        self.num_gpus = 1
        self.gather_noisy_modules()

    def _baseline_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=5e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.2, patience=3, mode="max"
            ),
            "monitor": "val_acc_top5",
            "interval": "epoch",
            "name": "lr",
        }
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return [optimizer], [scheduler]

    def _gs_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]
