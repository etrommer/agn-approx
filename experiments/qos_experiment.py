#!/usr/bin/env python3
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.ao.quantization as quant
import torchapprox.layers as tal
import torchapprox.utils.evoapprox as evo
from agnapprox.datamodules import CIFAR10, CIFAR100, MNIST, ApproxDataModule
from agnapprox.nets import ApproxNet, LeNet5, ResNet
from agnapprox.utils.select_multipliers import ApproximateMultiplier, select_multipliers
from experiment import ApproxExperiment
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class GradientSearchParams:
    lmbd: float
    sigma_max: float
    sigma_initial: float


class QoSExperiment(ApproxExperiment):
    def __init__(
        self,
        model: ApproxNet,
        datamodule: ApproxDataModule,
        mul_filter_str,
        qconfig: quant.QConfig,
        model_dir: str = "./models",
        test: bool = False,
    ) -> None:
        super().__init__(model, datamodule, model_dir, test)
        self.mul_filter_str = mul_filter_str
        self.qconfig = qconfig
        # Train quantized model during initialization
        # to provide

    def search_space(self) -> Dict[str, ApproximateMultiplier]:
        axmuls = {}
        for m in evo.module_names(self.mul_filter_str):
            axmuls[m] = ApproximateMultiplier(
                evo.error_map(m), float(evo.attribute(m, "PDK45_PWR"))
            )
        return axmuls

    def gradient_model(self, sigma_initial, sigma_max, lmbd) -> ApproxNet:
        grad_model = self.quantized_model(self.qconfig, "8ux8u_t")
        grad_path = os.path.join(
            self.model_dir,
            f"{self.model_fp32.name.lower()}_grad_{str(sigma_max).replace('.', '-')}_{str(lmbd).replace('.', '-')}.pt",
        )
        if not os.path.exists(grad_path):
            logger.debug(
                f"No gradient model for sigma_max={sigma_max}, lambda={sigma_max} found in {grad_path}. Training a new one."
            )
            grad_model.sigma_max = sigma_max
            for _, m in grad_model.approx_modules:
                m.stdev = sigma_initial
            grad_model.lmbd = lmbd
            mlf_params = {
                "lambda": lmbd,
                "sigma_max": sigma_max,
                "sigma_intial": sigma_initial,
            }
            pl.seed_everything(42)
            grad_model.train_noise(
                self.datamodule, mlf_params=mlf_params, test=self.test
            )
            torch.save(grad_model.state_dict(), grad_path)
        grad_model.load_state_dict(torch.load(grad_path))
        grad_model.mode = "noise"
        if torch.cuda.is_available():
            grad_model.to("cuda")
        return grad_model

    def test_mul_config(
        self,
        luts: npt.NDArray,
        mlf_extra_params: Optional[Dict[str, Any]],
        mlf_artifacts: Optional[List[str]],
    ):
        model = self.quantized_model(self.qconfig, "8ux8u_t")
        print(f"Testing configuration: {luts}")
        if luts.shape[1] == 1:
            for lut, (_, m) in zip(luts, model.approx_modules):
                m.lut = lut[0]
        else:
            model.init_shadow_luts(luts)
        model.train_approx(
            self.datamodule,
            mlf_params=mlf_extra_params,
            mlf_artifacts=mlf_artifacts,
            test=self.test,
        )


def n_multiplier_search(
    experiment: QoSExperiment,
    mul_search_params: GradientSearchParams,
    n_multipliers: int,
    prune: bool = False,
):
    if prune:
        raise NotImplementedError("Pruning not implemented yet")

    SCALE_FACTORS = [0.3, 1.0, 2.0]

    # Sweep of lambda values:
    # Higher values = lower resource consumption, worse performance
    # Lower values = higher resource conumption, better performance
    model = experiment.gradient_model(
        mul_search_params.sigma_initial,
        mul_search_params.sigma_max,
        mul_search_params.lmbd,
    )
    layer_matching = select_multipliers(
        model, experiment.datamodule, experiment.search_space(), pl.Trainer()
    )
    # Extract result for current lambda value
    matching_results = pd.DataFrame(
        columns=[n for n in experiment.search_space().keys()],
        data=[layer.mul_stds for layer in layer_matching],
    )
    matching_results["layer"] = [layer.name for layer in layer_matching]
    matching_results["std_max"] = [layer.max_std for layer in layer_matching]
    matching_results["ops"] = [layer.opcount for layer in layer_matching]
    matching_results = pd.concat(
        [
            matching_results.assign(scale=(np.ones(len(matching_results)) * scale))
            for scale in SCALE_FACTORS
        ]
    )
    matching_results["std_max"] *= matching_results["scale"]
    EPS = 1e-6
    mul_cols = matching_results.columns.str.contains("mul8")
    matching_results.loc[:, mul_cols] /= (
        matching_results.std_max.values[:, np.newaxis] + EPS
    )
    sub_df = matching_results.loc[:, mul_cols]
    # Drop AMs that don't achieve acceptable accuracy anywhere
    sub_df = sub_df.loc[:, (sub_df <= 1.0).any()]
    # Transform points with stdev >= 1.0 using log-transform
    # This avoids a disproportionally strong drag of points that are not relevant to the final choice anyway
    sub_df[sub_df >= 1.0] = np.log(sub_df + EPS)

    # Append to dataframe with all results

    # k-Means Search Space:
    # Standard Deviation results across all multipliers for each layer for each lambda value
    # normalized to the respective layer's standard deviation
    kmeans = KMeans(n_clusters=n_multipliers, n_init=1).fit(sub_df.values)

    # Assign appropriate multiplier based on k-Means centroids
    def stdresults_to_mul(std_estimate, mul_names, mul_performance):
        df = pd.DataFrame(
            data=[std_estimate, mul_performance],
            columns=mul_names,
            index=["stds", "perf"],
        ).T
        return df[df.stds <= 1.0].perf.idxmin()

    mul_names = [sub_df.columns]
    mul_performance = [
        experiment.search_space()[m].performance_metric for m in sub_df.columns
    ]
    mul_choice = np.array(
        [
            stdresults_to_mul(cc, mul_names, mul_performance)
            for cc in kmeans.cluster_centers_
        ]
    )
    matching_results["assignment"] = mul_choice[kmeans.labels_]

    # Calculate power savings
    matching_results["pwr_factor"] = [
        evo.attribute(n, "PDK45_PWR") / max(mul_performance)
        for n in matching_results.assignment
    ]

    # Test each configuration
    luts = []
    log_params = {"n_multipliers": n_multipliers}
    for i, s in enumerate(np.unique(matching_results["scale"])):
        current_df = matching_results[matching_results["scale"] == s]
        power_reduction = (
            current_df.pwr_factor * (current_df.ops / current_df.ops.sum())
        ).sum()
        log_params[f"power_reduction_{i}"] = power_reduction
        log_params[f"scale_{i}"] = s
        luts.append(
            np.array(
                [
                    np.load(f"/home/elias/evo_luts/{m}.npy")
                    for m in current_df.assignment
                ]
            )
        )

    for l in luts:
        lut = np.transpose(np.array(l)[np.newaxis, :], (1, 0, 2, 3))
        with tempfile.TemporaryDirectory() as tmpdirname:
            df_path = os.path.join(tmpdirname, "matching_result.csv")
            matching_results.to_csv(df_path)
            experiment.test_mul_config(lut, log_params, [df_path])


qconfig_8ux8u = quant.QConfig(
    activation=quant.FakeQuantize.with_args(
        observer=quant.MovingAverageMinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
    ),
    weight=quant.FakeQuantize.with_args(
        observer=quant.MinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
    ),
)


def lenet_mnist():
    net = LeNet5()
    net.name = "QoS_LeNet5"

    dm = MNIST(batch_size=128, num_workers=4)
    experiment = QoSExperiment(net, dm, "mul8u", qconfig=qconfig_8ux8u)

    mul_search_params = GradientSearchParams([0.5, 0.05, 0.005], 1.0, 0.5)
    n_multipliers = 3

    n_multiplier_search(experiment, mul_search_params, n_multipliers)


def resnet_cifar10():
    parameters = [
        # ("ResNet8", 4, GradientSearchParams([0.3], 0.3, 0.1)),
        # ("ResNet14", 4, GradientSearchParams([0.3], 0.15, 0.05)),
        ("ResNet20", 3, GradientSearchParams([0.2], 0.075, 0.01)),
        ("ResNet32", 3, GradientSearchParams([0.15], 0.075, 0.01)),
    ]
    for size, n_multipliers, mul_search_params in parameters:
        net = ResNet(resnet_size=size)
        net.name = f"QoS_{size}"

        dm = CIFAR10(batch_size=128, num_workers=4)
        experiment = QoSExperiment(net, dm, "mul8u", qconfig=qconfig_8ux8u, test=True)

        # fixed_mul = "mul8u_197B"
        # pwr_factor = experiment.search_space()[fixed_mul].performance_metric / max(
        #     [m.performance_metric for m in experiment.search_space().values()]
        # )
        # experiment.test_mul_config(
        #     [fixed_mul]
        #     * len(experiment.quantized_model(qconfig_8ux8u, "8ux8u_t").approx_modules),
        #     {"multiplier": fixed_mul, "power_reduction": pwr_factor},
        #     [],
        # )

        n_multiplier_search(experiment, mul_search_params, n_multipliers)


def resnet_cifar100():
    parameters = [
        # ("ResNet8", 4, GradientSearchParams([0.3], 0.3, 0.1)),
        # ("ResNet14", 4, GradientSearchParams([0.3], 0.15, 0.05)),
        # ("ResNet20", 3, GradientSearchParams([0.001], 0.005, 0.001)),
        ("ResNet32", 3, GradientSearchParams([0.001], 0.005, 0.001)),
    ]
    for size, n_multipliers, mul_search_params in parameters:
        net = ResNet(resnet_size=size, num_classes=100)
        net.name = f"QoS_{size}_cifar100"
        net.topk = (1, 5)

        dm = CIFAR100(batch_size=128, num_workers=4)
        experiment = QoSExperiment(net, dm, "mul8u", qconfig=qconfig_8ux8u, test=True)
        n_multiplier_search(experiment, mul_search_params, n_multipliers)


if __name__ == "__main__":
    # lenet_mnist()
    # resnet_cifar10()
    resnet_cifar100()
