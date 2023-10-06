import logging
import os

# Silence warnings
import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from agnapprox.datamodules import CIFAR10
from agnapprox.nets import AlexNet

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.DEBUG)

pl.seed_everything(42, workers=True)


def get_baseline_model(path: str) -> pl.LightningModule:
    pl.seed_everything(42, workers=True)
    model = AlexNet(num_classes=10, deterministic=True)
    model.load_state_dict(torch.load(path))
    model.to(torch.device("cuda"))
    return model


def main():
    dm = CIFAR10(batch_size=128, num_workers=16)
    dm.prepare_data()
    dm.setup()

    # multipliers = module_names("mul8s")
    multipliers = []
    size = "AlexNet"
    path = os.path.join("ref_model_{}.pt".format(size.lower()))

    for mul in multipliers:
        # Train Baseline & quantized
        if not os.path.exists(path):
            model = AlexNet(num_classes=10, deterministic=True)
            model.train_baseline(dm, test=True)
            model.train_quant(dm, test=True)
            torch.save(model.state_dict(), path)
        else:
            model = get_baseline_model(path, size)

        # Baseline Accuracy
        for _, m in model.noisy_modules:
            # m.approx_op.lut = lut(mul)
            m.approx_op.lut = np.zeros((256, 256))
        model.train_approx(dm, name_ext=f" - Baseline - {mul}", epochs=0, test=True)
        del model
        torch.cuda.empty_cache()

        # Noise Accuracy
        model = get_baseline_model(path, size)
        # noise = estimate_noise(model, dm, lut(mul))
        noise = 0
        for (mean, stdev), (name, module) in zip(noise, model.noisy_modules):
            module.mean = mean
            module.stdev = stdev
        model.train_noise(dm, name_ext=f" - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # HTP Model accuracy
        model = get_baseline_model(path, size)
        for _, m in model.noisy_modules:
            # m.approx_op.lut = lut(mul)
            m.approx_op.lut = np.zeros((256, 256))
            # m.fast_model = htp_models_mul8s[mul]
        model.train_approx(dm, name_ext=f" - HTP - {mul}", test=True)
        del model
        torch.cuda.empty_cache()

        # Behavioral Sim
        model = get_baseline_model(path, size)
        for _, m in model.noisy_modules:
            # m.approx_op.lut = lut(mul)
            m.approx_op.lut = np.zeros((256, 256))
        model.train_approx(dm, name_ext=f" - Behavioral - {mul}", test=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
