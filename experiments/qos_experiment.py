#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import torch

from agnapprox.nets import ApproxNet
from agnapprox.datamodules import ApproxDataModule, MNIST
import torch.ao.quantization as quant

import torchapprox.utils.evoapprox as evo
from agnapprox.nets.lenet5 import LeNet5
from agnapprox.utils.select_multipliers import ApproximateMultiplier, select_multipliers
import pandas as pd
from sklearn.cluster import KMeans

import pytorch_lightning as pl
import numpy as np

from experiment import ApproxExperiment
from torchapprox.operators.htp_models.htp_models_mul8s import htp_models_mul8s


@dataclass
class GradientSearchParams:
    lmbd: Union[float, List[float]]
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

    def search_space(self) -> Dict[str, ApproximateMultiplier]:
        axmuls = {}
        for m in evo.module_names(self.mul_filter_str):
            axmuls[m] = ApproximateMultiplier(
                evo.error_map(m), float(evo.attribute(m, "PDK45_PWR"))
            )
        return axmuls

    def gradient_model(self, sigma_initial, sigma_max, lmbd) -> ApproxNet:
        model = self.baseline_model
        model.sigma_max = sigma_max
        for _, m in model.approx_modules:
            m.stdev = sigma_initial
        model.lmbd = lmbd
        mlf_params = {
            "lambda": lmbd,
            "sigma_max": sigma_max,
            "sigma_intial": sigma_initial,
        }
        model.train_noise(self.datamodule, mlf_params=mlf_params, test=self.test)
        return model

    def test_mul_config(
        self, multipliers: List[str], mlf_extra_params: Optional[Dict[str, Any]]
    ):
        model = self.quantized_model(self.qconfig, "8ux8u_t")
        print(f"Testing configuration: {multipliers}")
        for mul_name, (_, m) in zip(multipliers, model.approx_modules):
            m.lut = np.load(f"/home/elias/evo_luts/{mul_name}.npy")
        model.train_approx(self.datamodule, mlf_params=mlf_extra_params, test=self.test)


def n_multiplier_search(
    experiment: QoSExperiment,
    mul_search_params: GradientSearchParams,
    n_multipliers: int,
    prune: bool = False,
):
    if prune:
        raise NotImplementedError("Pruning not implemented yet")

    matching_results = pd.DataFrame()
    # Sweep of lambda values:
    # Higher values = lower resource consumption, worse performance
    # Lower values = higher resource conumption, better performance
    for lmbd in mul_search_params.lmbd:
        model = experiment.gradient_model(
            mul_search_params.sigma_initial, mul_search_params.sigma_max, lmbd
        )
        matching_result = select_multipliers(
            model, experiment.datamodule, experiment.search_space(), pl.Trainer()
        )
        # Extract result for current lambda value
        layer_result = pd.DataFrame(
            columns=[n for n in experiment.search_space().keys()],
            data=[layer.mul_stds for layer in matching_result],
        )
        layer_result["layer"] = [layer.name for layer in matching_result]
        layer_result["std_max"] = [layer.max_std for layer in matching_result]
        layer_result["ops"] = [layer.opcount for layer in matching_result]
        layer_result["lambda"] = lmbd
        # Append to dataframe with all results
        matching_results = pd.concat(
            [matching_results, layer_result], ignore_index=True
        )

    # k-Means Search Space:
    # Standard Deviation results across all multipliers for each layer for each lambda value
    # normalized to the respective layer's standard deviation
    EPS = 1e-6
    std_results = matching_results.loc[
        :, matching_results.columns.str.contains("mul8")
    ].values / (matching_results.std_max.values[:, np.newaxis] + EPS)
    # Transform points with stdev >= 1.0 using log-transform
    # This avoids a disproportionally strong drag of points that are not relevant to the final choice anyway
    std_results = np.where(
        std_results >= 1.0, 1 + np.log(std_results + EPS), std_results
    )
    kmeans = KMeans(n_clusters=n_multipliers, n_init=1).fit(std_results)

    # Assign appropriate multiplier based on k-Means centroids
    def stdresults_to_mul(std_estimate, mul_names, mul_performance):
        df = pd.DataFrame(
            data=[std_estimate, mul_performance],
            columns=mul_names,
            index=["stds", "perf"],
        ).T
        return df[df.stds <= 1.0].perf.idxmin()

    mul_names = [n for n in experiment.search_space().keys()]
    mul_performance = [
        experiment.search_space()[m].performance_metric for m in mul_names
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
    for lmbd in np.unique(matching_results["lambda"]):
        current_df = matching_results[matching_results["lambda"] == lmbd]
        power_reduction = (
            current_df.pwr_factor * (current_df.ops / current_df.ops.sum())
        ).sum()
        log_params = {"power_reduction": power_reduction, "lambda": lmbd}
        experiment.test_mul_config(current_df.assignment.values, log_params)


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


if __name__ == "__main__":
    lenet_mnist()
