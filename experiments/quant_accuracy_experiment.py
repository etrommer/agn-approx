#!/usr/bin/env python3

import experiment
import torch
import torch.ao.quantization as quant
from agnapprox.datamodules import MNIST
from agnapprox.datamodules.approx_datamodule import ApproxDataModule
from agnapprox.nets import LeNet5
from agnapprox.nets.approxnet import ApproxNet
from torchapprox import layers as tal
import numpy as np
from glob import glob

# with open('/home/elias/evo_luts/mul8x2u_0TD.npy', 'rb') as f:
#     lut = np.load(f)

BITWIDTH = 2
multipliers = glob(f"/home/elias/evo_luts/mul8x{BITWIDTH}u_*.npy")


class QuantAccuracyExperiment(experiment.ApproxExperiment):
    def __init__(
        self,
        model: ApproxNet,
        datamodule: ApproxDataModule,
        name: str,
        model_dir: str = "./models",
        test: bool = False,
    ) -> None:
        super().__init__(model, datamodule, name, model_dir, test)

    def test_qconfig(self, qconfig: quant.QConfig, qtype: str, lut_path: str):
        mul_name = lut_path.split("/")[-1].split(".")[0]
        mlf_extra_params = {
            "mul_name": mul_name,
            "qtype": qtype,
            "experiment": "quant_comparison",
            "gradient_clip": 0.5,
        }
        lut = np.load(lut_path)
        qmodel = self.quantized_model(qconfig, qtype)
        for n, m in qmodel.approx_modules:
            m.approx_op.lut = lut
        qmodel.train_approx(
            self.datamodule,
            test=self.test,
            gradient_clip_val=0.5,
            mlf_params=mlf_extra_params,
        )


weight_quant_configs = [
    (
        quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            quant_min=0,
            quant_max=2**BITWIDTH - 1,
        ),
        f"8x{BITWIDTH}_affine_per_tensor",
    ),
    (
        quant.FakeQuantize.with_args(
            observer=quant.MovingAveragePerChannelMinMaxObserver,
            dtype=torch.quint8,
            qscheme=torch.per_channel_affine,
            quant_min=0,
            quant_max=2**BITWIDTH - 1,
        ),
        f"8x{BITWIDTH}_affine_per_channel",
    ),
]


def lenet_mnist():
    net = LeNet5()
    dm = MNIST(batch_size=128, num_workers=4)
    experiment = QuantAccuracyExperiment(net, dm, "LeNet5", test=True)
    default_qconfig = tal.ApproxLayer.default_qconfig()

    for mul in multipliers:
        for wq, wqname in weight_quant_configs:
            qconfig = quant.QConfig(activation=default_qconfig.activation, weight=wq)
            experiment.test_qconfig(qconfig, wqname, mul)


if __name__ == "__main__":
    lenet_mnist()
