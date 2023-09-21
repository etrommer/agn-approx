"""
Class definition for AlexNet Approximate NN
"""
import logging

import torch
import torchvision

from .approxnet import ApproxNet

logger = logging.getLogger(__name__)


# pylint: disable=too-many-ancestors
class SqueezeNet(ApproxNet):
    """
    Definition of training hyperparameters for
    approximate AlexNet
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.name = "SqueezeNet"
        self.model = torchvision.models.squeezenet1_1(
            torchvision.models.SqueezeNet1_1_Weights.DEFAULT
        )

        # Replace last layer with randomly initialized layer of correct size
        if num_classes != 1000:
            self.model.classifier[1] = torch.nn.Conv2d(
                512, num_classes, kernel_size=(1, 1)
            )
            self.model.num_classes = num_classes
            self.model.classifier[0] = torch.nn.BatchNorm2d(512)

        self.topk = (1, 5)
        self.epochs: dict = {
            "baseline": 35,
            "qat": 8,
            "noise": 2,
            "approx": 2,
        }
        self.num_gpus = 1
        self.convert_layers()

    def _baseline_optimizers(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        for n, p in self.model.named_parameters():
            if "alpha" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=5e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4)
        return [optimizer], [scheduler]

    def _approx_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return [optimizer], [scheduler]

    def _gs_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]
