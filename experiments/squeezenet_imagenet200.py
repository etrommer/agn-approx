#!/usr/bin/env python3
import os
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)
import logging
import warnings

import torch
from torchapprox.utils.evoapprox import lut, module_names
from torchapprox.operators.htp_models.htp_models_mul8s import htp_models_mul8s

from agnapprox.datamodules import TinyImageNet
from agnapprox.nets import SqueezeNet
from agnapprox.utils.select_multipliers import estimate_noise

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.DEBUG)


def get_baseline_model(path: str) -> pl.LightningModule:
    pl.seed_everything(42, workers=True)
    model = SqueezeNet(deterministic=True)
    model.load_state_dict(torch.load(path))
    model.to(torch.device("cuda"))
    return model


def main():
    dm = TinyImageNet(batch_size=128, num_workers=16)
    dm.prepare_data()
    dm.setup()

    multipliers = module_names("mul8s")
    size = "SqueezeNet"
    path = os.path.join("ref_model_{}.pt".format(size.lower()))

    for mul in multipliers:
        # Train Baseline & quantized
        if not os.path.exists(path):
            model = SqueezeNet(deterministic=True)
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
        torch.cuda.empty_cache()

        # Noise Accuracy
        model = get_baseline_model(path, size)
        noise = estimate_noise(model, dm, lut(mul))
        for (mean, stdev), (name, module) in zip(noise, model.noisy_modules):
            module.mean = mean
            module.stdev = stdev
        model.train_noise(dm, name_ext=f" - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # HTP Model accuracy
        model = get_baseline_model(path, size)
        for _, m in model.noisy_modules:
            m.approx_op.lut = lut(mul)
            m.fast_model = htp_models_mul8s[mul]
        model.train_approx(dm, name_ext=f" - HTP - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # Behavioral Sim
        model = get_baseline_model(path, size)
        for _, m in model.noisy_modules:
            m.approx_op.lut = lut(mul)
        model.train_approx(dm, name_ext=f" - Behavioral - {mul}", test=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
