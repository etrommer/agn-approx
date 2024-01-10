#!/usr/bin/env python3
import os

import numpy as np
import pytorch_lightning as pl


# Silence warnings
import warnings

import torch
from torchapprox.operators.htp_models.htp_models_log import htp_models_log

from agnapprox.datamodules import MNIST
from agnapprox.nets import LeNet5
from agnapprox.utils.select_multipliers import estimate_noise

pl.seed_everything(42, workers=True)
# from torchapprox.utils.evoapprox import lut, module_names


warnings.filterwarnings("ignore", category=DeprecationWarning)
# logging.basicConfig(level=logging.DEBUG)

dummy_lut = np.zeros((256, 256))
k = 5


def get_baseline_model(path: str, size: str, mul: str) -> pl.LightningModule:
    pl.seed_everything(42, workers=True)
    model = LeNet5()
    model.load_state_dict(torch.load(path))
    model.deterministic = True
    model.to(torch.device("cuda"))
    model.lut = dummy_lut
    return model


def main():
    dm = MNIST(data_dir="/home/elias/agn_approx/data", batch_size=256, num_workers=24)
    dm.prepare_data()
    dm.setup()

    # multipliers = module_names("mul8s")
    size = "LeNet5_rev"
    path = os.path.join("ref_model_lenet5.pt")

    if not os.path.exists(path):
        # Train Baseline and quantized baseline
        model = LeNet5()
        model.train_baseline(dm, log_mlflow=True, test=True)
        model.train_quant(dm, log_mlflow=True, test=True)
        torch.save(model.state_dict(), path)

    for mul in [f"mitch_trunc_{k}"]:
        # Baseline Accuracy
        model = get_baseline_model(path, size, mul)
        for _, m in model.noisy_modules:
            m.fast_model = htp_models_log["accurate"]
        model.train_approx(dm, name_ext=f" - Baseline - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # Noise Accuracy
        model = get_baseline_model(path, size, mul)
        noise = estimate_noise(model, dm, dummy_lut)
        for (mean, stdev), (name, module) in zip(noise, model.noisy_modules):
            module.mean = -mean
            module.stdev = stdev
        model.train_noise(dm, name_ext=f" - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # Linear regression Model accuracy
        model = get_baseline_model(path, size, mul)
        for _, m in model.noisy_modules:
            m.fast_model = htp_models_log[f"lin_mitch_trunc_{k}"]
        model.train_approx(dm, name_ext=f" - Linear - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # HTP Model accuracy
        model = get_baseline_model(path, size, mul)
        for _, m in model.noisy_modules:
            m.fast_model = htp_models_log["htp_mitchell_trunc"]
        model.train_approx(dm, name_ext=f" - HTP - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # Behavioral Sim
        model = get_baseline_model(path, size, mul)
        for _, m in model.noisy_modules:
            m.approx_op.lut = dummy_lut
        model.train_approx(dm, name_ext=f" - Behavioral - {mul}", test=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
