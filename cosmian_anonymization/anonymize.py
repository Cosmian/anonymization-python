# -*- coding: utf-8 -*-
import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from humps import decamelize

from .conversion_helper import convert_config_types
from .method_parser import create_transformation_function
from .noise_correlation import NoiseCorrelationTask, parse_noise_correlation_config


def create_output_dataframe(df: pd.DataFrame, config: Dict, inplace: bool):
    """Create an output DataFrame based on a configuration.
    Check that the input data match the configuration provided.

    Args:
        df (pd.DataFrame): The input DataFrame.
        config (Dict): The configuration specifying the desired output columns and their types.
        inplace (bool): Whether to modify the input DataFrame in-place or create a new DataFrame.

    Returns:
        pd.DataFrame: The output DataFrame with columns in the specified order.

    Raises:
        ValueError: If a column specified in the configuration is missing from the input DataFrame.
    """
    output_df = df
    if not inplace:
        output_df = pd.DataFrame()

    # List of the column names in the config's order
    sorted_output_columns = []
    for column_metadata in config["metadata"]:
        col_name = column_metadata["name"]
        col_type = column_metadata["type"]

        if col_name not in df:
            # Column missing from the dataset
            raise ValueError(f"Missing column from data: {col_name}.")

        if "method" in column_metadata and column_metadata["method"] == "DeleteColumn":
            continue  # Do not add column to the output dataframe

        # Add this column as output to match the config's order
        sorted_output_columns.append(col_name)

        # Make sure the type of the pandas column is the same as the one in the config
        output_df[col_name] = convert_config_types(df[col_name], col_type)

    # Return the output dataframe with columns in the config's order
    return output_df[sorted_output_columns]


def apply_anonymization_column(
    values: pd.Series,
    method: Optional[str] = None,
    method_options: Dict = {},
):
    """Apply anonymization to a specific column in a DataFrame.

    Args:
        values (pd.Series): The values to apply anonymization to.
        name (str): The name of the column to anonymize.
        method (Optional[str], optional): The method to use for anonymization. Defaults to None.
        method_options (Dict, optional): Additional options for the anonymization method. Defaults to {}.
        **kwargs: Additional config fields are ignored.

    Returns:
        pd.Series: The anonymized column.

    Raises:
        ValueError: If the specified column is missing from the DataFrame.

    Notes:
        - If `method` is None, the original column is returned without applying any anonymization.
        - In case of noise with correlation, the function returns None.

    """
    if method is None or "correlation" in method_options:
        # No method to apply for this column
        return values

    if "correlation" in method_options:
        # Correlation is done later in a dedicated function: `apply_correlation_columns`
        return values

    # Create a transformation function based on the selected technique.
    transform_func = create_transformation_function(method, method_options)

    result: List[Any] = []
    # Apply anonymization to each element
    for i, val in enumerate(values):
        try:
            result.append(transform_func(val))
        except Exception as e:
            raise ValueError(f"Error processing `{values.name}` at line {i + 1}:\n{e}")

    return result


def apply_correlation_columns(values: np.ndarray, task: NoiseCorrelationTask):
    """Apply noise correlation to specified columns in a DataFrame.

    Args:
        values (np.ndarray): The values to apply noise correlation to, in the shape (lines, columns).
        task (NoiseCorrelationTask): The task containing column names and transformation function.

    Returns:
        pd.DataFrame: The columns with noise correlation applied.
    """
    transform_func = task.generate_transformation()

    result: List[Any] = []
    # Apply correlated anonymization to both columns line by line
    for i, line_values in enumerate(values):
        try:
            result.append(transform_func(line_values))
        except Exception as e:
            raise ValueError(
                f"Error processing `{task.column_names}` at line {i + 1}:\n{e}"
            )

    return result


def anonymize_dataframe(
    df: pd.DataFrame, config: Dict, inplace: bool = False
) -> pd.DataFrame:
    """
    Anonymizes a Pandas DataFrame by applying the specified techniques to selected columns.

    Args:
        df: The input DataFrame to anonymize.
        config: A dictionary containing the metadata for each column to anonymize.
        inplace: If True, applies the anonymization directly to the input DataFrame.
            If False, creates a new DataFrame with the anonymized data.
    """
    # Convert config from camel case to snake case
    config = decamelize(config)

    # Init output dataframe to match the config columns
    output_df = create_output_dataframe(df, config, inplace)

    # Iterate over each column to anonymize.
    for column_metadata in config["metadata"]:
        col_name: str = column_metadata["name"]

        if col_name not in output_df:
            continue  # Column has been deleted

        # Anonymize the column
        output_df[col_name] = apply_anonymization_column(
            output_df[col_name],
            column_metadata.get("method", None),
            column_metadata.get("method_options", {}),
        )

    # -- Noise correlation --
    # Read through the config to find all correlation tasks
    noise_corr_tasks = parse_noise_correlation_config(config)
    # Apply correlation on each groups
    for task in noise_corr_tasks.values():
        output_df[task.column_names] = apply_correlation_columns(
            output_df[task.column_names].values, task
        )

    # Return the anonymized data
    return output_df


def anonymize_from_files(data_path: str, config_path: str, output_path: str) -> None:
    """
    Reads the configuration file and data file, anonymizes the data according to the configuration,
    and writes the anonymized data to a new file.

    Args:
        data_path (str): The path to the CSV file containing the data to anonymize.
        config_path (str): The path to the configuration file.
        output_path (str): The path where to write the anonymized data as a CSV file.
    """

    # Read the configuration file and convert keys to snake_case.
    with open(config_path, "r") as f:
        config = json.load(f)

    df = pd.read_csv(data_path, sep=config["configurationInfo"]["delimiter"])

    try:
        # Anonymize the data according to the configuration.
        anonymized_df = anonymize_dataframe(df, config)
    except ValueError as e:
        print("Anonymization failed:", e)
        return

    # Write the anonymized data to the output file.
    anonymized_df.to_csv(
        output_path, sep=config["configurationInfo"]["delimiter"], index=False
    )
    print(f"Anonymized data written to {output_path}.")


def cli():
    """
    Command-line interface for Cosmian Data Anonymization.

    This function parses the command-line arguments and calls the anonymization function.

    Usage:
        cosmian-anonymize <input_csv> <input_config> <output_csv>
    """
    # Create an argument parser with a description
    parser = argparse.ArgumentParser(
        description="Command-line interface for Cosmian Data Anonymization."
    )

    # Add the required arguments
    parser.add_argument(
        "input_csv", type=str, help="Path to the data to anonymize in CSV format."
    )
    parser.add_argument(
        "input_config", type=str, help="Path to the configuration file."
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Path where to store the anonymized data in CSV format.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the anonymization function with the provided arguments
    anonymize_from_files(args.input_csv, args.input_config, args.output_csv)
