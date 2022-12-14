#!/usr/bin/env python3
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

from agnapprox.datamodules import TinyImageNet
from agnapprox.nets import VGG
from agnapprox.utils.select_multipliers import estimate_noise

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.DEBUG)


def get_baseline_model(path: str, size: str) -> pl.LightningModule:
    pl.seed_everything(42, workers=True)
    model = VGG(vgg_size=size, deterministic=True)
    model.load_state_dict(torch.load(path))
    model.to(torch.device("cuda"))
    return model


def main():
    dm = TinyImageNet(
        data_dir="/home/elias/agn_approx/data", batch_size=128, num_workers=8
    )
    dm.prepare_data()
    dm.setup()

    multipliers = module_names("mul8s")
    size = "VGG16"
    path = os.path.join("ref_model_{}_gs.pt".format(size.lower()))

    for mul in multipliers:

        # Train Baseline & quantized
        model = VGG(vgg_size=size)
        if not os.path.exists(path):
            # Train Baseline and quantized baseline
            model.train_baseline(dm, log_mlflow=True)
            model.train_quant(dm, log_mlflow=True)
            torch.save(model.state_dict(), path)
        else:
            model.load_state_dict(torch.load(path))
            model.to(torch.device("cuda"))

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
