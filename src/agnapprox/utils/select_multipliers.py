"""
Utility functions to select approximate multipliers based on reference data
"""
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl

import agnapprox.utils.error_stats as stats
from agnapprox.utils.model import get_feature_maps

if TYPE_CHECKING:
    from agnapprox.nets import ApproxNet
    from agnapprox.utils.model import IntermediateLayerResults


logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """
    Multiplier Matching result for a single layer
    """

    name: str
    opcount: float
    max_std: float
    mul_stds: List[float]


@dataclass
class ApproximateMultiplier:
    error_map: npt.NDArray
    performance_metric: float


def select_layer_multiplier(
    intermediate_results: "IntermediateLayerResults",
    info: LayerInfo,
    multipliers: Dict[str, ApproximateMultiplier],
    max_noise: float,
    num_samples: int = 512,
) -> LayerInfo:
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
    max_std = (
        np.std(intermediate_results.features @ intermediate_results.weights) * max_noise
    )
    # logger.warning(
    #     "Layer Standard Deviation: %f Maximum Standard Deviation: %f",
    #     np.std(intermediate_results.features @ intermediate_results.weights),
    #     max_std,
    # )

    info.max_std = max_std

    for mul_name, mul in multipliers.items():
        # Calculate error standard deviation for current multiplier
        _, mul_std = stats.population_prediction(mul.error_map, x_dists, w_dist, fan_in)
        # Error is calculated w.r.t. to a single multiplication and
        # reduced standard deviation from fan-in is compensated.
        # We need to scale it to the numerical range of the neuron output to make
        # it comparable.
        mul_std *= fan_in
        info.mul_stds.append(mul_std)

    return info


def select_multipliers(
    model: "ApproxNet",
    datamodule: pl.LightningDataModule,
    multipliers: Dict[str, ApproximateMultiplier],
    trainer: pl.Trainer,
) -> List[LayerInfo]:
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
    ref_data = get_feature_maps(model, model.approx_modules, trainer, datamodule)

    result = []
    for name, module in model.approx_modules:
        layer_result = LayerInfo(name, module.opcount, 0, [])
        layer_result = select_layer_multiplier(
            ref_data[name], layer_result, multipliers, abs(module.stdev.item())
        )
        result.append(layer_result)

    return result
