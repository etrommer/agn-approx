"""
Utility functions to select approximate multipliers based on reference data
"""
import logging
from typing import Any, Dict, TYPE_CHECKING

import agnapprox.utils.error_stats as stats
import numpy as np
import pytorch_lightning as pl
from agnapprox.utils.model import get_feature_maps

if TYPE_CHECKING:
    from agnapprox.nets import ApproxNet

logger = logging.getLogger(__name__)


def select_layer_multiplier(
    layer_ref_data: Dict[str, Any],
    multipliers: Dict[str, Any],
    max_noise: float,
    num_samples: int = 512,
) -> Dict[str, Any]:
    """
    Select a matching approximate multiplier for a single layer

    Args:
        layer_ref_data: Dictionary of Reference input/output data generated from
            a model run with accurate multiplication
        multipliers: Approximate Multiplier Error Maps, Performance Metrics and name
        max_noise: Learned allowable noise parameter (sigma_l)
        num_samples: _description_. Defaults to 512.

    Returns:
        Dictionary with name and performance metric of selected multiplier
    """
    fan_in = layer_ref_data["fan_in"]

    # Create weight and input probability distributions
    sample_pop = stats.get_sample_population(
        layer_ref_data["input"], num_samples=num_samples
    )
    x_dists = np.array([stats.to_distribution(x, -128, 127)[1] for x in sample_pop])
    _, w_dist = stats.to_distribution(layer_ref_data["weights"], -128, 127)

    # Maximum tolerable standard deviation
    max_std = np.std(layer_ref_data["output"]) * max_noise
    logger.debug(
        "Layer Standard Deviation: %f Maximum Standard Deviation: %f",
        np.std(layer_ref_data["output"]),
        max_std,
    )

    metric_max = max([m["metric"] for m in multipliers]) + 1.0
    best_match = {"metric": metric_max}

    for mul in multipliers:
        # Calculate error standard deviation for current multiplier
        _, mul_std = stats.population_prediction(mul["emap"], x_dists, w_dist, fan_in)
        # Error is calculated w.r.t. to a single multiplication and
        # reduced standard deviation from fan-in is compensated.
        # We need to scale it to the numerical range of the neuron output to make
        # it comparable.
        mul_std *= fan_in
        logger.debug(
            "Multiplier %s: Standard Deviation: %f, Metric: %f",
            mul["name"],
            mul_std,
            mul["metric"],
        )

        # Check if multiplier is within tolerance and improves the performance metric
        if mul_std <= max_std and mul["metric"] <= best_match["metric"]:
            best_match = mul
            best_match["standard_deviation"] = mul_std

    logger.debug(
        "Best Match: %s, Metric: %f, SF: %f",
        best_match["name"],
        best_match["metric"],
        best_match["metric"] / (metric_max - 1.0),
    )
    return {"name": best_match["name"], "metric": best_match["metric"]}


def select_multipliers(
    model: 'ApproxNet',
    datamodule: pl.LightningDataModule,
    library,
    trainer: pl.Trainer,
    signed: bool = True,
    deploy: bool = False,
):
    """
    Select matching Approximate Multipliers for all layers in a model

    Args:
        model: Approximate Model with learned layer robustness parameters
        datamodule: Data Module to use for sampling runs
        library: Approximate Multiplier Library provider
        trainer: PyTorch Lightning Trainer instance to use for sampling run
        signed: Whether to select signed or unsigned instances from Multiplier library provide.
            Defaults to True.
        deploy: Whether to write selected approximate multiplier to layer configuration.
            Defaults to False.

    Returns:
        Dictionary of Assignment results
    """
    multipliers = library.prepare(signed=signed)
    ref_data = get_feature_maps(model, model.noisy_modules, trainer, datamodule)

    result = {"metric_max": max([m["metric"] for m in multipliers])}
    for name, module in model.noisy_modules:
        logger.debug("Matching Multiplier to Layer %s", name)
        result[name] = select_layer_multiplier(
            ref_data[name], multipliers, abs(module.stdev.item())
        )
        result[name]["saving_factor"] = result[name]["metric"] / result["metric_max"]
        result[name]["opcount"] = module.opcount.item()
        result[name]["relative_opcount"] = (module.opcount / model.total_ops).item()

        if deploy:
            module.approx_op.lut = library.load_lut(result[name]["name"])
    result["total_saving_factor"] = sum(
        [
            l["saving_factor"] * l["relative_opcount"]
            for l in result.values()
            if isinstance(l, dict)
        ]
    )
    return result
