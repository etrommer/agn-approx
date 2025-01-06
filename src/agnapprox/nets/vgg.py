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
        if pretrained:
            weights = torchvision.models.VGG16_BN_Weights.DEFAULT

        self.name = vgg_size
        if self.name.lower() == "vgg11":
            self.model = torchvision.models.vgg11_bn(weights=weights)
        if self.name.lower() == "vgg13":
            self.model = torchvision.models.vgg13_bn(weights=weights)
        if self.name.lower() == "vgg16":
            self.model = torchvision.models.vgg16_bn(weights=weights)
        if self.name.lower() == "vgg19":
            self.model = torchvision.models.vgg19_bn(weights=weights)

        # Replace last layer with randomly initialized layer of correct size
        if num_classes != 1000:
            self.model.classifier[6] = torch.nn.Linear(4096, num_classes)

        self.topk = (1, 5)
        self.epochs: dict = {
            "baseline": 30,
            "qat": 8,
            "noise": 3,
            "approx": 2,
        }
        self.num_gpus = 1

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
        if self.tune_bn:
            params = []
            for p in self.parameters():
                p.requires_grad = False
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm1d) or isinstance(
                    m, torch.nn.BatchNorm2d
                ):
                    params.append(m.weight)
                    params.append(m.bias)
                    m.bias.requires_grad = True
                    m.weight.requires_grad = True
                    continue
                if hasattr(m, "bias") and m.bias is not None:
                    params.append(m.bias)
                    m.bias.requires_grad = True
        else:
            params = [p for p in self.parameters()]
        optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return [optimizer], [scheduler]

    def _noise_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]
