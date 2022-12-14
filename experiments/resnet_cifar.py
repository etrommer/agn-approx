#!/usr/bin/env python3
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
from torchapprox.operators.htp_models.htp_models_mul8s import htp_models_mul8s
from torchapprox.utils.evoapprox import lut, module_names

from agnapprox.datamodules import CIFAR10
from agnapprox.nets import ResNet
from agnapprox.utils.select_multipliers import estimate_noise

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.DEBUG)


def get_baseline_model(path: str, size: str) -> pl.LightningModule:
    pl.seed_everything(42, workers=True)
    model = ResNet(resnet_size=size, deterministic=True)
    model.load_state_dict(torch.load(path))
    model.to(torch.device("cuda"))
    return model


def main():
    dm = CIFAR10(batch_size=128, num_workers=16)
    dm.prepare_data()
    dm.setup()

    multipliers = module_names("mul8s")
    size = "ResNet8"
    path = os.path.join("ref_model_{}.pt".format(size.lower()))

    for mul in multipliers:
        # Train Baseline & quantized
        if not os.path.exists(path):
            model = ResNet(resnet_size=size, deterministic=True)
            model.train_baseline(dm, test=True)
            model.train_quant(dm, test=True)
            torch.save(model.state_dict(), path)
        else:
            model = get_baseline_model(path, size)

        # Baseline Accuracy
        for _, m in model.noisy_modules:
            m.approx_op.lut = lut(mul)
        model.train_approx(dm, name_ext=f" - Baseline - {mul}", epochs=0, test=True)
        del model

        # Noise Accuracy
        model = get_baseline_model(path, size)
        noise = estimate_noise(model, dm, lut(mul))
        for (mean, stdev), (name, module) in zip(noise, model.noisy_modules):
            module.mean = mean
            module.stdev = stdev
        model.train_noise(dm, name_ext=f" - {mul}", test=True)
        del model

        # HTP Model accuracy
        model = get_baseline_model(path, size)
        for _, m in model.noisy_modules:
            m.approx_op.lut = lut(mul)
            m.fast_model = htp_models_mul8s[mul]
        model.train_approx(dm, name_ext=f" - HTP - {mul}", test=True)
        del model

        # Behavioral Sim
        model = get_baseline_model(path, size)
        for _, m in model.noisy_modules:
            m.approx_op.lut = lut(mul)
        model.train_approx(dm, name_ext=f" - Behavioral - {mul}", test=True)
        del model


if __name__ == "__main__":
    main()
