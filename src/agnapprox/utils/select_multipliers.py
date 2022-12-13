"""
Utility functions to select approximate multipliers based on reference data
"""
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl

import agnapprox.utils.error_stats as stats
from agnapprox.libs.approxlib import ApproxLibrary
from agnapprox.utils.model import get_feature_maps
import torch

if TYPE_CHECKING:
    from torchapprox.utils.evoapprox import ApproximateMultiplier

    from agnapprox.nets import ApproxNet
    from agnapprox.utils.model import IntermediateLayerResults


logger = logging.getLogger(__name__)


def select_layer_multiplier(
    intermediate_results: "IntermediateLayerResults",
    multipliers: List["ApproximateMultiplier"],
    max_noise: float,
    num_samples: int = 512,
) -> Tuple[str, float]:
    """
    Select a matching approximate multiplier for a single layer

    Args:
        layer_ref_data: Reference input/output data generated from
            a model run with *accurate* multiplication. This is used to calibrate
            the layer standard deviation for the error estimate and determine the
            distribution of numerical values in weights and activations.
        multipliers: Approximate Multiplier Error Maps, Performance Metrics and name
        max_noise: Learned allowable noise parameter (sigma_l)
        num_samples: Number of samples to draw from features for multi-population prediction.
            Defaults to 512.

    Returns:
        Dictionary with name and performance metric of selected multiplier
    """
    fan_in = intermediate_results.fan_in

    # Create weight and input probability distributions
    sample_pop = stats.get_sample_population(
        intermediate_results.features, num_samples=num_samples
    )
    x_dists = np.array([stats.to_distribution(x, -128, 127)[1] for x in sample_pop])
    _, w_dist = stats.to_distribution(intermediate_results.weights, -128, 127)

    # Maximum tolerable standard deviation
    max_std = np.std(intermediate_results.outputs) * max_noise
    logger.debug(
        "Layer Standard Deviation: %f Maximum Standard Deviation: %f",
        np.std(intermediate_results.outputs),
        max_std,
    )

    @dataclass
    class Match:
        """
        Container that tracks the best AM seen so far in the search space
        """

        performance_metric: float
        stdev: float = 0.0
        idx: Optional[int] = None

    # Initialize best match to be worse than anything in the search space
    metric_max = max([m.performance_metric for m in multipliers]) + 1e-3
    best_match = Match(metric_max)

    for idx, mul in enumerate(multipliers):
        # Calculate error standard deviation for current multiplier
        _, mul_std = stats.population_prediction(mul.error_map, x_dists, w_dist, fan_in)
        # Error is calculated w.r.t. to a single multiplication and
        # reduced standard deviation from fan-in is compensated.
        # We need to scale it to the numerical range of the neuron output to make
        # it comparable.
        mul_std *= fan_in

        # Check if multiplier is within accuracty tolerance and
        # improves the performance metric
        if (
            mul_std <= max_std
            and mul.performance_metric <= best_match.performance_metric
        ):
            best_match = Match(mul.performance_metric, mul_std, idx)

        logger.debug(
            "Multiplier %s: Standard Deviation: %f, Metric: %f",
            mul.name,
            mul_std,
            mul.performance_metric,
        )

    if best_match.idx is None:
        raise ValueError(
            "Search did not yield any result. Possibly empty search space?"
        )
    result = multipliers[best_match.idx]
    return result.name, result.performance_metric


@dataclass
class LayerInfo:
    """
    Multiplier Matching result for a single layer
    """

    name: str
    multiplier_name: str
    multiplier_performance_metric: float
    opcount: float

    def relative_opcount(self, total_opcount: float):
        """
        Calculate the relative contribution of this layer to the network's total operations

        Args:
            total_opcount: Number of operations in the entire networks

        Returns:
            float between 0..1 where:
            - 0: layer contributes no operations to the network's opcount
            - 1: layer conttibutes all operations to the network's opcount
        """
        return self.opcount / total_opcount

    def relative_energy_consumption(self, metric_max: float):
        """
        Relative energy consumption of selected approximate multiplier

        Args:
            metric_max: Highest possible value for performance metric
                (typically that of the respective accurate multiplier)

        Returns:
            float between 0..1 where:
            - 0: selected multiplier consumes no energy
            - 1: selected multiplier consumes the maximum amount of energy
        """
        return self.multiplier_performance_metric / metric_max


@dataclass
class MatchingInfo:
    """
    Multiplier Matching result for the entire model
    """

    layers: List[LayerInfo]
    metric_max: float
    opcount: float

    @property
    def relative_energy_consumption(self):
        """
        Relative Energy Consumption compared to network without approximation
        achieved by the current AM configuration

        Returns:
            sum relative energy consumption for each layer,
            weighted with the layer's contribution to overall operations
        """
        return sum(
            [
                l.relative_opcount(self.opcount)
                * l.relative_energy_consumption(self.metric_max)
                for l in self.layers
            ]
        )


def deploy_multipliers(
    model: "ApproxNet", matching_result: MatchingInfo, library: ApproxLibrary
):
    """
    Deploy selected approximate multipliers to network

    Args:
        model: Model to deploy multipliers to
        matching_result: Results of multiplier matching
    """
    for layer_info, (name, module) in zip(matching_result.layers, model.noisy_modules):
        assert (
            layer_info.name == name
        ), "Inconsistent layer order between model and optimization results"
        module.approx_op.lut = library.load_lut(layer_info.multiplier_name)


def estimate_noise(
    model: "ApproxNet",
    datamodule: pl.LightningDataModule,
    lut: np.ndarray,
) -> List[Tuple[float, float]]:

    for _, m in model.noisy_modules:
        m.approx_op.lut = None

    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=1)
    ref_data = get_feature_maps(model, model.noisy_modules, trainer, datamodule)

    ans = []
    for ref, (_, m) in zip(ref_data.values(), model.noisy_modules):
        with torch.no_grad():
            features = torch.from_numpy(ref.features).to(m.weight.device)
            m.approx_op.lut = lut
            approx = m(features).detach().cpu().numpy()
        error = ref.outputs - approx
        ans.append((np.mean(error), np.std(error) / np.std(ref.outputs)))

    return ans


def select_multipliers(
    model: "ApproxNet",
    datamodule: pl.LightningDataModule,
    multipliers: List["ApproximateMultiplier"],
    trainer: pl.Trainer,
) -> MatchingInfo:
    """
    Select matching Approximate Multipliers for all layers in a model

    Args:
        model: Approximate Model with learned layer robustness parameters
        datamodule: Data Module to use for sampling runs
        library: Approximate Multiplier Library provider
        trainer: PyTorch Lightning Trainer instance to use for sampling run
        signed: Whether to select signed or unsigned instances from Multiplier library provide.
            Defaults to True.

    Returns:
        Dictionary of Assignment results
    """
    ref_data = get_feature_maps(model, model.noisy_modules, trainer, datamodule)

    metric_max = max([m.performance_metric for m in multipliers])
    result = MatchingInfo([], metric_max, model.total_ops.item())
    for name, module in model.noisy_modules:
        mul_name, mul_metric = select_layer_multiplier(
            ref_data[name], multipliers, abs(module.stdev.item())
        )
        layer_result = LayerInfo(name, mul_name, mul_metric, module.opcount.item())
        result.layers.append(layer_result)

        logger.info(
            "Layer: %s, Best Match: %s, Performance: %f, Relative Performance: %f",
            name,
            mul_name,
            mul_metric,
            mul_metric / metric_max,
        )

    return result
