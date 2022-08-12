"""
Implementations for several probabilistic error models
"""

from typing import Optional, Tuple

import numpy as np


def single_dist_mc(
    emap: np.ndarray,
    x_dist: np.ndarray,
    w_dist: np.ndarray,
    fan_in: float,
    num_samples: int = int(1e5),
) -> Tuple[float, float]:
    """
    Generate error mean and standard deviation using Monte Carlo
    approach as described in: https://arxiv.org/abs/1912.00700

    Args:
        emap: The multiplier's error map
        x_dist: Operand distribution of activations
        w_dist: Operand distribution of weights
        fan_in: Incoming connections for layer
        num_samples: Number of Monte Carlo simulation runs. Defaults to int(1e5).

    Returns:
        Mean and standard deviation for a single operation
    """
    prob_x, prob_w = np.meshgrid(
        x_dist.astype(np.float64), w_dist.astype(np.float64), indexing="ij"
    )
    probabilities = (prob_x * prob_w).flatten()
    emap = emap.flatten()
    monte_carlo_runs = np.random.choice(
        emap, size=(num_samples, fan_in), p=probabilities
    )
    monte_carlo_runs = np.sum(monte_carlo_runs, axis=1)
    return (
        np.mean(monte_carlo_runs) / fan_in,
        np.std(monte_carlo_runs, dtype=np.float64) / fan_in,
    )


def error_prediction(
    emap: np.ndarray, x_dist: np.ndarray, w_dist: np.ndarray, fan_in: float
) -> Tuple[float, float]:
    """
    Generate error mean and standard deviation using the
    global distribution of activations and weights

    Args:
        emap: The multiplier's error map
        x_dist: Operand distribution of activations
        w_dist: Operand distribution of weights
        fan_in: Incoming connections for layer

    Returns:
        Mean and standard deviation for a single operation
    """
    emap = emap.astype(np.float64)
    prob_x, prob_w = np.meshgrid(
        x_dist.astype(np.float64), w_dist.astype(np.float64), indexing="ij"
    )
    mean = np.sum(emap * prob_x * prob_w)
    std = np.sqrt(np.sum(((emap - mean) ** 2) * prob_x * prob_w)) / np.sqrt(fan_in)
    return mean, std


def get_sample_population(tensor: np.ndarray, num_samples: int = 512) -> np.ndarray:
    """
    Randomly select samples from a tensor that cover the receptive field of one neuron

    Args:
        tensor: Tensor to draw samples from
        num_samples: Number of samples to draw. Defaults to 512.

    Returns:
        Sampled 2D Tensor of shape [num_samples, tensor.shape[-1]]
    """
    flat_dim = np.prod(tensor.shape[:-1])
    rand_idx = np.random.choice(flat_dim, num_samples)
    return tensor.reshape(flat_dim, tensor.shape[-1])[rand_idx]


def population_prediction(
    emap: np.ndarray, x_multidist: np.ndarray, w_dist: np.ndarray, fan_in: float
) -> Tuple[float, float]:
    """
    Generate prediction of mean and standard deviation using several
    sampled local distributions

    Args:
        emap: The multiplier's error map
        x_multidist: Array of several operand distributions for activations
        w_dist: Operand distribution of weights
        fan_in: Incoming connections for layer

    Returns:
        Mean and standard deviation for a single operation
    """
    # Single distribution error computation for each operand distribution
    means, stds = [], []
    for x_dist in x_multidist:
        mean, std = error_prediction(emap, x_dist, w_dist, fan_in)
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)

    # Aggregate error distributions (Eq. 15 & Eq. 16)
    mean_aggregate = np.mean(means)
    std_aggregate = np.sqrt(
        (np.sum(means ** 2 + stds ** 2) - (np.sum(means) ** 2) / x_multidist.shape[0])
        / x_multidist.shape[0]
    )
    return mean_aggregate, std_aggregate


def to_distribution(
    tensor: Optional[np.ndarray], min_val: int, max_val: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn tensor of weights/activations into a frequency distribution (i.e. build a histogram)

    Args:
        tensor: Tensor to build histogram from
        min_val: Lowest possible operand value in tensor
        max_val: Highest possible operand value in tensor

    Returns:
        Tuple of Arrays where first array contains the full numerical range between
        min_val and max_val inclusively and second array contains the relative frequency
        of each operand

    Raises:
        ValueError: If run before features maps have been populated 
        by call to `utils.model.get_feature_maps`

    """
    if tensor is None:
        raise ValueError("Populate input tensor with intermediate features maps")
    num_range = np.arange(min_val, max_val + 1)
    counts = np.zeros_like(num_range)
    nums, freqs = np.unique(tensor.flatten().astype(np.int32), return_counts=True)
    counts[nums + min_val] = freqs.astype(np.float64)
    counts = counts / np.sum(freqs)
    return num_range, counts


def error_calculation(
    accurate: np.ndarray, approximate: np.ndarray, fan_in: float
) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation of the observed error between
    accurate computation and approximate computation

    Args:
        accurate: Accurate computation results
        approximate: Approximate computation results
        fan_in: Number of incoming neuron connections

    Returns:
        Mean and standard deviation for a single operation
    """
    mean = np.mean(accurate - approximate) / fan_in
    std = np.std((accurate - approximate) / fan_in, dtype=np.float64)
    return mean, std
