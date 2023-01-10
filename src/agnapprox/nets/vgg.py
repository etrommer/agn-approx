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
        **kwargs
    ):
        super().__init__(**kwargs)

        self.name = vgg_size
        if self.name == "VGG11":
            self.model = torchvision.models.vgg11_bn(
                weights=torchvision.models.VGG11_BN_Weights.DEFAULT
            )
        if self.name == "VGG13":
            self.model = torchvision.models.vgg13_bn(
                weights=torchvision.models.VGG13_BN_Weights.DEFAULT
            )
        if self.name == "VGG16":
            self.model = torchvision.models.vgg16_bn(
                weights=torchvision.models.VGG16_BN_Weights.DEFAULT
            )
        if self.name == "VGG19":
            self.model = torchvision.models.vgg19_bn(
                weights=torchvision.models.VGG19_BN_Weights.DEFAULT
            )

        # Replace last layer with randomly initialized layer of correct size
        if num_classes != 1000:
            self.model.classifier[6] = torch.nn.Linear(4096, num_classes)

        self.topk = (1, 5)
        self.epochs: dict = {
            "baseline": 30,
            "qat": 2,
            "noise": 1,
            "approx": 1,
        }
        self.num_gpus = 1
        self.gather_noisy_modules()

    def _baseline_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
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
        optimizer = torch.optim.SGD(
            self.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):

        # Freeze BatchNorm layers
        # for m in self.model.modules():
        #     if isinstance(m, torch.nn.BatchNorm2d) or isinstance(
        #         m, torch.nn.BatchNorm1d
        #     ):
        #         m.eval()

        # Freeze feature detector params
        for p in self.parameters():
            p.requires_grad = False

        # Train Classifier params
        trainable_params = []
        for n, p in self.model.classifier.named_parameters():
            p.requires_grad = True
            trainable_params.append(p)

        optimizer = torch.optim.SGD(trainable_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return [optimizer], [scheduler]

    def _gs_optimizers(self):
        pass
