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

    def __init__(self, num_classes: int = 200, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.name = "MobileNetV2"
        self.model = torchvision.models.mobilenet_v2(
            torchvision.models.MobileNet_V2_Weights.DEFAULT
        )

        # Replace last layer with randomly initialized layer of correct size
        if num_classes != 1000:
            self.model.classifier[1] = torch.nn.Linear(1280, num_classes)

        self.topk = (1, 5)
        self.epochs: dict = {
            "baseline": 30,
            "qat": 8,
            "noise": 2,
            "approx": 2,
        }
        self.num_gpus = 1
        self.convert_layers()

    def _baseline_optimizers(self):
        # for p in self.model.parameters():
        #     p.requires_grad = False
        # self.model.classifier[1].requires_grad = True
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-2)
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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4
        )
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
