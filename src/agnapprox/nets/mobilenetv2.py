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
            "qat": 20,
            "noise": 5,
            "prune": 20,
            "approx": 2,
        }
        self.pruning_epochs = int(self.epochs["prune"] * 0.8)
        self.model.features[0][0].stride = (1, 1)
        self.num_gpus = 1

    def _baseline_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[18, 26])
        return [optimizer], [scheduler]

    def _qat_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-2, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8)
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
        optimizer = torch.optim.SGD(params=params, lr=2e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return [optimizer], [scheduler]

    def _noise_optimizers(self):
        params = [m.stdev for _, m in self.approx_modules]
        optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        return [optimizer], [scheduler]

    def _prune_optimizers(self):
        return self._qat_optimizers()
