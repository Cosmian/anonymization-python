# -*- coding: utf-8 -*-
import argparse
import json
from typing import Dict, Optional

import pandas as pd
from humps import decamelize

from .method_parser import create_transformation_function
from .noise_correlation import NoiseCorrelationTask, parse_noise_correlation_config


def apply_anonymization_column(
    df: pd.DataFrame,
    name: str,
    method: Optional[str] = None,
    method_options: Dict = {},
    **kwargs,
):
    """Apply anonymization to a specific column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to apply anonymization to.
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
    if name not in df:
        # Column missing from the dataset
        raise ValueError(f"Missing column from data: {name}.")

    if method is None:
        # No method to apply for this column
        return df[name]

    if "correlation" in method_options:
        # Correlation is done later in a dedicated function: `apply_correlation_columns`
        return None

    # Create a transformation function based on the selected technique.
    transform_func = create_transformation_function(method, method_options)
    return df[name].map(transform_func)


def apply_correlation_columns(df: pd.DataFrame, task: NoiseCorrelationTask):
    """Apply noise correlation to specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to apply noise correlation to.
        task (NoiseCorrelationTask): The task containing column names and transformation function.

    Returns:
        pd.DataFrame: The columns with noise correlation applied.
    """
    transform_func = task.generate_transformation()
    return df[task.column_names].apply(transform_func, axis=1, raw=True)


# TODO: extract code
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

    sorted_output_columns = []
    anonymized_df = df
    if not inplace:
        anonymized_df = pd.DataFrame()

    # Iterate over each column to anonymize.
    for column_metadata in config["metadata"]:
        col_name: str = column_metadata["name"]
        # Add this column as output to match the config's order
        sorted_output_columns.append(col_name)

        anonymized_column = apply_anonymization_column(df, **column_metadata)
        # Return column could be None for correlation methods
        if anonymized_column is not None:
            anonymized_df[col_name] = anonymized_column

    # -- Noise correlation --
    # Read through the config to find all correlation tasks
    noise_corr_tasks = parse_noise_correlation_config(config)
    # Apply correlation on each groups
    for task in noise_corr_tasks.values():
        anonymized_df[task.column_names] = apply_correlation_columns(df, task)

    # Return the anonymized data with columns in the config's order
    return anonymized_df[sorted_output_columns]


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
    anonymized_df.to_csv(output_path, sep=";", index=False)
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
