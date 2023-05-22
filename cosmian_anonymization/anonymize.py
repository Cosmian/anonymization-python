# -*- coding: utf-8 -*-
import json
from typing import Dict

import pandas as pd
from humps import decamelize
from method_parser import create_transformation_function
from noise_correlation import parse_noise_correlation_config


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
        if col_name not in df:
            # Column missing from the dataset
            raise ValueError(f"Missing column from data: {col_name}.")
        # Add this column as output to match the config's order
        sorted_output_columns.append(col_name)

        if "method" not in column_metadata:
            # No method to apply for this column
            anonymized_df[col_name] = df[col_name]
            continue
        method_name: str = column_metadata["method"]
        method_opts: Dict = (
            column_metadata["method_options"]
            if "method_options" in column_metadata
            else {}
        )
        if "correlation" in method_opts:
            # Skip correlation for now
            continue
        # Create a transformation function based on the selected technique.
        transform_func = create_transformation_function(method_name, method_opts)
        anonymized_df[col_name] = df[col_name].map(transform_func)

    # Noise correlation

    # Read through the config to find all correlation tasks
    noise_corr_tasks = parse_noise_correlation_config(config)
    # Apply correlation on each groups
    for correlation_task in noise_corr_tasks.values():
        transform_func = correlation_task.generate_transformation()
        anonymized_df[correlation_task.column_names] = df[
            correlation_task.column_names
        ].apply(transform_func, axis=1, raw=True)

    # Return the anonymized data with columns in the config's order
    return anonymized_df[sorted_output_columns]


def anonymize(config_path: str, data_path: str, output_path: str) -> None:
    """
    Reads the configuration file and data file, anonymizes the data according to the configuration,
    and writes the anonymized data to a new file.

    Args:
        config_path (str): The path to the configuration file.
        data_path (str): The path to the data file.
        output_path (str): The path to the output file.
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


if __name__ == "__main__":
    anonymize(
        "./tests/data/config-correlated.json",
        "./tests/data/data-correlated.csv",
        "./tests/data/out.csv",
    )
