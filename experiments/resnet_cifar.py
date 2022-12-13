import copy
import os

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)

import logging

# Silence warnings
import warnings

import torch
from torchapprox.utils.evoapprox import lut, module_names

from agnapprox.datamodules import CIFAR10
from agnapprox.nets import ResNet
from agnapprox.utils.select_multipliers import estimate_noise

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.DEBUG)


def main():
    dm = CIFAR10(batch_size=128, num_workers=16)
    dm.prepare_data()
    dm.setup()

    multipliers = module_names("mul8s")
    size = "ResNet8"

    for mul in multipliers:
        path = os.path.join("ref_model_{}.pt".format(size.lower()))

        if not os.path.exists(path):
            # Train Baseline and quantized baseline
            model = ResNet(resnet_size=size, deterministic=True)
            model.train_baseline(dm, log_mlflow=True)
            model.train_quant(dm, log_mlflow=True)
            torch.save(model.state_dict(), path)
        else:
            model = ResNet(resnet_size=size, deterministic=True)
            model.load_state_dict(torch.load(path))
            model.to(torch.device("cuda"))

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=1,
        )
        noise = estimate_noise(model, dm, trainer, lut(mul))
        for (mean, stdev), (name, module) in zip(noise, model.noisy_modules):
            module.mean = mean
            module.stdev = stdev

        model.train_noise(dm, log_mlflow=True, name_ext=f" - {mul}")

        del model


if __name__ == "__main__":
    main()
