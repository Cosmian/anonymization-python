# -*- coding: utf-8 -*-
from typing import Dict, Optional

from cloudproof_anonymization import NoiseGenerator

from .conversion_helper import DURATION_IN_SECONDS


def create_noise_generator(
    distribution: str,
    mean: Optional[float] = None,
    std_dev: Optional[float] = None,
    lower_boundary: Optional[float] = None,
    upper_boundary: Optional[float] = None,
) -> NoiseGenerator:
    """
    Returns a `NoiseGenerator` object based on the specified options.

    Args:
        method (str): The distribution to use for generating noise: "Uniform", "Gaussian", or "Laplace".
        mean (float, optional): The mean value to use for generating noise if `option_type` is `params`.
        std_dev (float, optional): The standard deviation value to use for generating noise if `option_type` is `params`.
        min_bound (float, optional): The minimum value to use for generating noise if `option_type` is `bounds`.
        max_bound (float, optional): The maximum value to use for generating noise if `option_type` is `bounds`.
    """
    if mean is not None and std_dev is not None:
        return NoiseGenerator.new_with_parameters(distribution, mean, std_dev)
    elif lower_boundary is not None and upper_boundary is not None:
        return NoiseGenerator.new_with_bounds(
            distribution, lower_boundary, upper_boundary
        )
    else:
        raise ValueError("Missing noise options.")


def create_date_noise_generator(
    distribution: str,
    mean: Optional[Dict] = None,
    std_dev: Optional[Dict] = None,
    lower_boundary: Optional[Dict] = None,
    upper_boundary: Optional[Dict] = None,
) -> NoiseGenerator:
    """
    Returns a `NoiseGenerator` object based on the specified options.

    Args:
        distribution (str): A string indicating the distribution to use for generating noise.
        mean (float, optional): The mean value to use for generating noise.
        std_dev (Dict, optional): A dictionary with the following keys:
            - precision (float): The precision value for the noise generator.
            - unit (str): A string indicating the unit of time for the noise generator.
        min_bound (Dict, optional):
            - precision (float): The precision value for the minimum bound.
            - unit (str): A string indicating the unit of time for the minimum bound.
        max_bound (Dict, optional):
            - precision (float): The precision value for the maximum bound.
            - unit (str): A string indicating the unit of time for the maximum bound.
    """
    mean_secs: Optional[float] = None
    std_dev_secs: Optional[float] = None
    if mean is not None and std_dev is not None:
        # Convert mean and standard deviation to seconds
        mean_secs = mean["precision"] * DURATION_IN_SECONDS[mean["unit"]]
        std_dev_secs = std_dev["precision"] * DURATION_IN_SECONDS[std_dev["unit"]]

    lower_boundary_secs: Optional[float] = None
    upper_boundary_secs: Optional[float] = None
    if lower_boundary is not None and upper_boundary is not None:
        # Convert range to seconds
        lower_boundary_secs = (
            lower_boundary["precision"] * DURATION_IN_SECONDS[lower_boundary["unit"]]
        )
        upper_boundary_secs = (
            upper_boundary["precision"] * DURATION_IN_SECONDS[upper_boundary["unit"]]
        )

    return create_noise_generator(
        distribution,
        mean=mean_secs,
        std_dev=std_dev_secs,
        lower_boundary=lower_boundary_secs,
        upper_boundary=upper_boundary_secs,
    )
