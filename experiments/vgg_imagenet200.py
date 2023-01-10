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
from torchapprox.operators.htp_models.htp_models_mul8s import htp_models_mul8s
from torchapprox.utils.evoapprox import lut, module_names

from agnapprox.datamodules import TinyImageNet
from agnapprox.nets import VGG
from agnapprox.utils.select_multipliers import estimate_noise

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.DEBUG)


def get_baseline_model(path: str, size: str, mul: str) -> pl.LightningModule:
    pl.seed_everything(42, workers=True)
    model = VGG(vgg_size=size, num_classes=200)
    model.load_state_dict(torch.load(path))
    model.deterministic = True
    model.to(torch.device("cuda"))
    model.lut = lut(mul)
    # new_nm = [(n, m) for (n, m) in model.noisy_modules if "classifier" in n]
    # model.noisy_modules = new_nm
    return model


def main():
    dm = TinyImageNet(
        data_dir="/home/elias/agn_approx/data", batch_size=16, num_workers=24
    )
    dm.prepare_data()
    dm.setup()

    multipliers = module_names("mul8s")
    size = "VGG16"
    path = os.path.join("ref_model_{}.pt".format(size.lower()))

    if not os.path.exists(path):
        # Train Baseline and quantized baseline
        model = VGG(vgg_size=size, num_classes=200, deterministic=False)
        model.train_baseline(dm, log_mlflow=True, test=True)
        model.train_quant(dm, log_mlflow=True, test=True)
        torch.save(model.state_dict(), path)

    for mul in multipliers:
        if not "1L2L" in mul and not "1L2N" in mul:
            continue
        # Baseline Accuracy
        model = get_baseline_model(path, size, mul)
        for _, m in model.noisy_modules:
            m.fast_model = htp_models_mul8s["accurate"]
        model.train_approx(dm, name_ext=f" - Baseline - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # Noise Accuracy
        model = get_baseline_model(path, size, mul)
        noise = estimate_noise(model, dm, lut(mul))
        for (mean, stdev), (name, module) in zip(noise, model.noisy_modules):
            module.mean = -mean
            module.stdev = stdev
        model.train_noise(dm, name_ext=f" - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # HTP Model accuracy
        model = get_baseline_model(path, size, mul)
        for _, m in model.noisy_modules:
            m.fast_model = htp_models_mul8s[mul]
        model.train_approx(dm, name_ext=f" - HTP - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # Behavioral Sim
        model = get_baseline_model(path, size, mul)
        for _, m in model.noisy_modules:
            m.approx_op.lut = lut(mul)
        model.train_approx(dm, name_ext=f" - Behavioral - {mul}", test=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
