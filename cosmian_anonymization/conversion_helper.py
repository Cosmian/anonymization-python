# -*- coding: utf-8 -*-
from datetime import timezone
from typing import Any, Dict

import pandas as pd
from dateutil import parser as date_parser

# Mapping between the configuration types and Python types
CONFIG_TYPES_MAPPING: Dict[str, Any] = {
    "Text": "string",
    "Date": "string",
    "Integer": "int64",
    "Float": "float64",
}

# Time duration units in seconds
DURATION_IN_SECONDS: Dict[str, int] = {
    "Second": 1,
    "Minute": 60,
    "Hour": 3600,
    "Day": 86400,
    "Month": 2_628_000,
    "Year": 31_536_000,
}


def convert_config_types(values: pd.Series, type: str) -> pd.Series:
    """Convert a pandas Series to a specified type.

    Args:
        series (pd.Series): The pandas Series to convert.
        type (str): The target type to convert the series to.

    Returns:
        pd.Series: The converted pandas Series.

    """
    if type not in CONFIG_TYPES_MAPPING:
        raise ValueError(f"Invalid type in config file: {type}.")

    target_type = CONFIG_TYPES_MAPPING[type]

    # Convert the series to the target type if it's different from the current type
    try:
        converted_values = (
            values.astype(target_type) if values.dtype != target_type else values
        )

        if type == "Date":
            # Convert the series values to RFC3339 format
            converted_values = converted_values.map(date_to_rfc3339)

    except Exception:
        raise ValueError(
            f"The column `{values.name}` contains elements that could not be converted to {type}."
        )

    return converted_values


def date_to_rfc3339(date_str: str) -> str:
    """
    Converts a date string to ISO format with timezone (RFC 3339).

    Args:
        date_str (str): The input date string.

    Returns:
        str: The date string in RFC3339 format.
    """
    dt = date_parser.parse(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
