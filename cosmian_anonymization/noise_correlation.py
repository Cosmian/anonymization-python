# -*- coding: utf-8 -*-
from typing import Callable, Dict, List

from .noise_parser import create_date_noise_generator, create_noise_generator


class NoiseCorrelationTask:
    """
    Class representing a noise correlation task.
    """

    def __init__(self, method: str, opts: Dict):
        """
        Initialize a NoiseCorrelationTask.

        Args:
            method (str): The noise distribution.
            opts (Dict): Options for the noise generation.
        """
        self.method = method
        # The keywords `correlation` and `fine_tuning` are not used to create the generator
        self.options = {
            key: value
            for key, value in opts.items()
            if key != "correlation" and key != "fine_tuning"
        }
        self.column_names: List[str] = []

    def add_column(self, column_name: str):
        """
        Add a column name to the list of column names.

        Args:
            column_name (str): The column name to add.
        """
        self.column_names.append(column_name)

    def generate_transformation(self) -> Callable[[List], List]:
        """
        Generate and return the transformation function for applying correlated noise.

        Returns:
            Callable[[List], List]: The transformation function.
        """
        # Mapping of noise method to noise generator functions
        noise_func_mapping: Dict[str, Callable] = {
            "NoiseDate": lambda **kwargs: create_date_noise_generator(
                **kwargs
            ).apply_correlated_noise_on_dates,
            "NoiseInteger": lambda **kwargs: create_noise_generator(
                **kwargs
            ).apply_correlated_noise_on_ints,
            "NoiseFloat": lambda **kwargs: create_noise_generator(
                **kwargs
            ).apply_correlated_noise_on_floats,
        }
        # Get the noise generator function based on the specified method
        noise_generator_func = noise_func_mapping.get(self.method)
        if noise_generator_func is None:
            raise ValueError(f"Cannot apply correlation for method: {self.method}.")

        # Scale noise by 1 for now
        correlation_factors = [1] * len(self.column_names)
        # Create a noise generator instance with the specified options
        noise_generator = noise_generator_func(**self.options)

        # Return a function that applies the noise generator to the data vector
        def apply_noise(data_vec: List) -> List:
            return noise_generator(data_vec, correlation_factors)

        return apply_noise


def parse_noise_correlation_config(config: Dict) -> Dict[str, NoiseCorrelationTask]:
    """
    Parse the noise correlation configuration and return the dictionary of correlation tasks.

    Args:
        config (Dict): The noise correlation configuration.

    Returns:
        Dict[str, NoiseCorrelationTask]: The dictionary of correlation tasks.
    """
    tasks: Dict[str, NoiseCorrelationTask] = {}

    # Iterate over each column metadata in the configuration
    for column_metadata in config["metadata"]:
        col_name: str = column_metadata["name"]

        # Check if method and method_options are present
        if "method" not in column_metadata or "method_options" not in column_metadata:
            continue

        method_name: str = column_metadata["method"]
        method_opts = column_metadata["method_options"]

        # Check if correlation option is present
        if "correlation" not in method_opts:
            continue

        correlation_uid = method_opts["correlation"]

        # Create or retrieve the correlation task based on correlation_uid
        if correlation_uid not in tasks:
            tasks[correlation_uid] = NoiseCorrelationTask(method_name, method_opts)

        # Add the current column to the correlation task
        tasks[correlation_uid].add_column(col_name)

    return tasks
