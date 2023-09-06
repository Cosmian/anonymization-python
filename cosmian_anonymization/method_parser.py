# -*- coding: utf-8 -*-
import os
from typing import Callable, Dict, Optional

from cloudproof_py.anonymization import (
    DateAggregator,
    Hasher,
    NumberAggregator,
    NumberScaler,
    WordMasker,
    WordPatternMasker,
    WordTokenizer,
)
from cloudproof_py.fpe import Alphabet, Float, Integer

from .conversion_helper import date_to_rfc3339
from .noise_parser import create_date_noise_generator, create_noise_generator


def parse_date_aggregation_options(time_unit: str) -> Callable[[str], str]:
    """
    Parses the date aggregation options and returns a function that applies the aggregation.

    Args:
        time_unit (str): The time unit for rounding.
    """
    date_aggregator = DateAggregator(time_unit)

    def apply_date_aggregation(date_str: str) -> str:
        return date_aggregator.apply_on_date(date_to_rfc3339(date_str))

    return apply_date_aggregation


def parse_fpe_string_options(
    alphabet: str, extend_with: Optional[str] = None
) -> Callable[[str], str]:
    # TODO: get key and tweak from dedicated file
    fpe_string = Alphabet(alphabet)
    if extend_with:
        fpe_string.extend_with(extend_with)

    def fpe_string_encrypt(val: str) -> str:
        return fpe_string.encrypt(os.urandom(32), os.urandom(32), val)

    return fpe_string_encrypt


def parse_fpe_integer_options(radix: int, digit: int) -> Callable[[int], int]:
    # TODO: get key and tweak from dedicated file
    fpe_int = Integer(radix, digit)

    def fpe_int_encrypt(val: int) -> int:
        return fpe_int.encrypt(os.urandom(32), os.urandom(32), val)

    return fpe_int_encrypt


def parse_fpe_float_options() -> Callable[[float], float]:
    # TODO: get key and tweak from dedicated file
    fpe_float = Float()

    def fpe_float_encrypt(val: float) -> float:
        return fpe_float.encrypt(os.urandom(32), os.urandom(32), val)

    return fpe_float_encrypt


def parse_hash_options(
    hash_type: str, salt_value: Optional[str] = None, encoding="utf-8"
) -> Callable[[str], str]:
    """
    Returns a function that takes a string and applies a hash function to it.

    Args:
        hash_type (str): The name of the hash function to use.
        salt_value (Optional[str]): An optional salt to use for hashing.
        encoding (str): The encoding to use when converting the input string to bytes.
    """
    salt = None
    if salt_value:
        salt = salt_value.encode(encoding)

    return Hasher(hash_type, salt).apply_str


def create_transformation_function(method_name: str, method_opts: Dict) -> Callable:
    """
    Given a method name and options, returns a callable that applies the desired transformation.
    """
    parsing_functions: Dict[str, Callable] = {
        "FpeString": parse_fpe_string_options,
        "FpeInteger": parse_fpe_integer_options,
        "FpeFloat": parse_fpe_float_options,
        "TokenizeWords": lambda **kwargs: WordTokenizer(**kwargs).apply,
        "MaskWords": lambda **kwargs: WordMasker(**kwargs).apply,
        "Regex": lambda **kwargs: WordPatternMasker(**kwargs).apply,
        "Hash": parse_hash_options,
        "NoiseDate": lambda **kwargs: create_date_noise_generator(
            **kwargs
        ).apply_on_date,
        "NoiseInteger": lambda **kwargs: create_noise_generator(**kwargs).apply_on_int,
        "NoiseFloat": lambda **kwargs: create_noise_generator(**kwargs).apply_on_float,
        "AggregationDate": parse_date_aggregation_options,
        "AggregationInteger": lambda **kwargs: NumberAggregator(**kwargs).apply_on_int,
        "AggregationFloat": lambda **kwargs: NumberAggregator(**kwargs).apply_on_float,
        "RescalingInteger": lambda **kwargs: NumberScaler(**kwargs).apply_on_int,
        "RescalingFloat": lambda **kwargs: NumberScaler(**kwargs).apply_on_float,
    }
    parsing_function = parsing_functions.get(method_name)
    if parsing_function is None:
        raise ValueError(f"Unknown method named: {method_name}.")

    # remove "fine_tuning" field from method options
    filtered_method_opts = {
        key: method_opts[key] for key in method_opts if key != "fine_tuning"
    }
    return parsing_function(**filtered_method_opts)
