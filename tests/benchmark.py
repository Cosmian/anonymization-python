# -*- coding: utf-8 -*-
import argparse
import time
from typing import Any, Dict

import pandas as pd

from cosmian_anonymization import anonymize_dataframe

default_benchmarks = [
    {
        "name": "Hash (SHA3)",
        "sample_data": "Test",
        "n_samples": int(1e7),
        "data_type": "Text",
        "method": "Hash",
        "method_opts": {
            "hashType": "SHA3",
            "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
        },
    },
    {
        "name": "Hash (Argon2)",
        "sample_data": "Test",
        "n_samples": int(1e3),
        "data_type": "Text",
        "method": "Hash",
        "method_opts": {
            "hashType": "Argon2",
            "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
        },
    },
    {
        "name": "Noise Float (Gaussian)",
        "sample_data": 1.0,
        "n_samples": int(5e7),
        "data_type": "Float",
        "method": "NoiseFloat",
        "method_opts": {
            "distribution": "Gaussian",
            "lowerBoundary": -5.0,
            "upperBoundary": 5.0,
        },
    },
    {
        "name": "Noise Date (Gaussian)",
        "sample_data": "Jul 26, 2022",
        "n_samples": int(1e5),
        "data_type": "Date",
        "method": "NoiseDate",
        "method_opts": {
            "distribution": "Gaussian",
            "lowerBoundary": {"precision": 0, "unit": "Day"},
            "upperBoundary": {"precision": 5, "unit": "Day"},
        },
    },
    {
        "name": "Word Pattern Masking",
        "sample_data": "This person lived from 1810-04-01 to 1837-05-09.",
        "n_samples": int(1e7),
        "data_type": "Text",
        "method": "Regex",
        "method_opts": {"pattern": r"\b\d{4}-\d{2}-\d{2}\b", "replace": "DATE"},
    },
    {
        "name": "Word Tokenization",
        "sample_data": "The quick brown fox jumps over the lazy dog.",
        "n_samples": int(1e4),
        "data_type": "Text",
        "method": "TokenizeWords",
        "method_opts": {
            "wordsList": ["quick", "brown", "dog"],
        },
    },
    {
        "name": "Word Masking",
        "sample_data": "The quick brown fox jumps over the lazy dog.",
        "n_samples": int(1e4),
        "data_type": "Text",
        "method": "MaskWords",
        "method_opts": {
            "wordsList": ["quick", "brown", "dog"],
        },
    },
    {
        "name": "Number Aggregation (Float)",
        "sample_data": "123.456789",
        "n_samples": int(1e7),
        "data_type": "Float",
        "method": "AggregationFloat",
        "method_opts": {"powerOfTen": 2},
    },
    {
        "name": "Date Aggregation",
        "sample_data": "2023-04-27 16:23:45",
        "n_samples": int(1e5),
        "data_type": "Date",
        "method": "AggregationDate",
        "method_opts": {"timeUnit": "Day"},
    },
    {
        "name": "Number Scaling (Float)",
        "sample_data": "1.0",
        "n_samples": int(5e7),
        "data_type": "Float",
        "method": "RescalingFloat",
        "method_opts": {"mean": 0.0, "stdDev": 1.0, "scale": 2.0, "translation": 10.0},
    },
]


def benchmark_anonymize(
    name: str,
    sample_data: Any,
    n_samples: int,
    data_type: str,
    method: str,
    method_opts: Dict,
):
    print("==== Anonymization Benchmark ====")
    print("\tName:", name)
    print(f"\tN samples: {n_samples:,}")

    df = pd.DataFrame(
        {
            "input_text": [sample_data] * n_samples,
        }
    )

    config = {
        "metadata": [
            {
                "name": "input_text",
                "type": data_type,
                "method": method,
                "methodOptions": method_opts,
            },
        ],
    }

    start_time = time.time()
    anonymize_dataframe(df, config, inplace=True)
    processing_time = time.time() - start_time

    print(f"\tProcessing time: {processing_time:.2f}s")
    print(f"\tIterations per sec: {round(n_samples/processing_time):,}")


def bench(benchmarks):
    for bench_config in benchmarks:
        benchmark_anonymize(**bench_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process filters.")
    parser.add_argument(
        "--filter",
        help="Run specific tests (e.g. --filter hash,word,fpe,noise,aggregate,rescale)",
    )

    args = parser.parse_args()
    if not args.filter:
        bench(default_benchmarks)
        exit(0)

    filters = args.filter.split(",")

    if "hash" in filters:
        bench(
            [
                {
                    "name": "Hash (SHA2)",
                    "sample_data": "Test",
                    "n_samples": int(1e7),
                    "data_type": "Text",
                    "method": "Hash",
                    "method_opts": {
                        "hashType": "SHA2",
                        "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
                    },
                },
                {
                    "name": "Hash (SHA3)",
                    "sample_data": "Test",
                    "n_samples": int(1e7),
                    "data_type": "Text",
                    "method": "Hash",
                    "method_opts": {
                        "hashType": "SHA3",
                        "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
                    },
                },
                {
                    "name": "Hash (Argon2)",
                    "sample_data": "Test",
                    "n_samples": int(1e3),
                    "data_type": "Text",
                    "method": "Hash",
                    "method_opts": {
                        "hashType": "Argon2",
                        "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
                    },
                },
            ]
        )

    if "word" in filters:
        bench(
            [
                {
                    "name": "Word Pattern Masking",
                    "sample_data": "This person lived from 1810-04-01 to 1837-05-09.",
                    "n_samples": int(1e7),
                    "data_type": "Text",
                    "method": "Regex",
                    "method_opts": {
                        "pattern": r"\b\d{4}-\d{2}-\d{2}\b",
                        "replace": "DATE",
                    },
                },
                {
                    "name": "Word Tokenization",
                    "sample_data": "The quick brown fox jumps over the lazy dog.",
                    "n_samples": int(1e4),
                    "data_type": "Text",
                    "method": "TokenizeWords",
                    "method_opts": {
                        "wordsList": ["quick", "brown", "dog"],
                    },
                },
                {
                    "name": "Word Masking",
                    "sample_data": "The quick brown fox jumps over the lazy dog.",
                    "n_samples": int(1e4),
                    "data_type": "Text",
                    "method": "MaskWords",
                    "method_opts": {
                        "wordsList": ["quick", "brown", "dog"],
                    },
                },
            ]
        )

    if "fpe" in filters:
        bench(
            [
                {
                    "name": "FPE (String)",
                    "sample_data": "Michael",
                    "n_samples": int(5e5),
                    "data_type": "Text",
                    "method": "FpeString",
                    "method_opts": {"alphabet": "alpha"},
                },
                {
                    "name": "FPE (Integer)",
                    "sample_data": 12345,
                    "n_samples": int(5e5),
                    "data_type": "Integer",
                    "method": "FpeInteger",
                    "method_opts": {"radix": 10, "digit": 10},
                },
                {
                    "name": "FPE (Float)",
                    "sample_data": "12.345",
                    "n_samples": int(5e5),
                    "data_type": "Float",
                    "method": "FpeFloat",
                    "method_opts": {},
                },
            ]
        )

    if "aggregate" in filters:
        bench(
            [
                {
                    "name": "Number Aggregation (Float)",
                    "sample_data": "123.456789",
                    "n_samples": int(1e7),
                    "data_type": "Float",
                    "method": "AggregationFloat",
                    "method_opts": {"powerOfTen": 2},
                },
                {
                    "name": "Number Aggregation (Integer)",
                    "sample_data": "12345",
                    "n_samples": int(1e7),
                    "data_type": "Integer",
                    "method": "AggregationInteger",
                    "method_opts": {"powerOfTen": 2},
                },
                {
                    "name": "Date Aggregation",
                    "sample_data": "2023-04-27 16:23:45",
                    "n_samples": int(1e5),
                    "data_type": "Date",
                    "method": "AggregationDate",
                    "method_opts": {"timeUnit": "Day"},
                },
            ]
        )

    if "rescale" in filters:
        bench(
            [
                {
                    "name": "Number Scaling (Float)",
                    "sample_data": "1.0",
                    "n_samples": int(5e7),
                    "data_type": "Float",
                    "method": "RescalingFloat",
                    "method_opts": {
                        "mean": 0.0,
                        "stdDev": 1.0,
                        "scale": 2.0,
                        "translation": 10.0,
                    },
                },
                {
                    "name": "Number Scaling (Integer)",
                    "sample_data": "1",
                    "n_samples": int(5e7),
                    "data_type": "Integer",
                    "method": "RescalingInteger",
                    "method_opts": {
                        "mean": 0.0,
                        "stdDev": 1.0,
                        "scale": 2.0,
                        "translation": 10.0,
                    },
                },
            ]
        )

    if "noise" in filters:
        bench(
            [
                {
                    "name": "Noise Float (Gaussian)",
                    "sample_data": 1.0,
                    "n_samples": int(5e7),
                    "data_type": "Float",
                    "method": "NoiseFloat",
                    "method_opts": {
                        "distribution": "Gaussian",
                        "lowerBoundary": -5.0,
                        "upperBoundary": 5.0,
                    },
                },
                {
                    "name": "Noise Float (Laplace)",
                    "sample_data": 1.0,
                    "n_samples": int(5e7),
                    "data_type": "Float",
                    "method": "NoiseFloat",
                    "method_opts": {
                        "distribution": "Laplace",
                        "lowerBoundary": -5.0,
                        "upperBoundary": 5.0,
                    },
                },
                {
                    "name": "Noise Float (Uniform)",
                    "sample_data": 1.0,
                    "n_samples": int(5e7),
                    "data_type": "Float",
                    "method": "NoiseFloat",
                    "method_opts": {
                        "distribution": "Uniform",
                        "lowerBoundary": -5.0,
                        "upperBoundary": 5.0,
                    },
                },
                {
                    "name": "Noise Integer (Gaussian)",
                    "sample_data": 1,
                    "n_samples": int(5e7),
                    "data_type": "Integer",
                    "method": "NoiseInteger",
                    "method_opts": {
                        "distribution": "Gaussian",
                        "lowerBoundary": -5,
                        "upperBoundary": 5,
                    },
                },
                {
                    "name": "Noise Integer (Laplace)",
                    "sample_data": 1,
                    "n_samples": int(5e7),
                    "data_type": "Integer",
                    "method": "NoiseInteger",
                    "method_opts": {
                        "distribution": "Laplace",
                        "lowerBoundary": -5,
                        "upperBoundary": 5,
                    },
                },
                {
                    "name": "Noise Integer (Uniform)",
                    "sample_data": 1,
                    "n_samples": int(5e7),
                    "data_type": "Integer",
                    "method": "NoiseInteger",
                    "method_opts": {
                        "distribution": "Uniform",
                        "lowerBoundary": -5,
                        "upperBoundary": 5,
                    },
                },
                {
                    "name": "Noise Date (Gaussian)",
                    "sample_data": "Jul 26, 2022",
                    "n_samples": int(1e5),
                    "data_type": "Date",
                    "method": "NoiseDate",
                    "method_opts": {
                        "distribution": "Gaussian",
                        "lowerBoundary": {"precision": 0, "unit": "Day"},
                        "upperBoundary": {"precision": 5, "unit": "Day"},
                    },
                },
                {
                    "name": "Noise Date (Laplace)",
                    "sample_data": "Jul 26, 2022",
                    "n_samples": int(1e5),
                    "data_type": "Date",
                    "method": "NoiseDate",
                    "method_opts": {
                        "distribution": "Laplace",
                        "lowerBoundary": {"precision": 0, "unit": "Day"},
                        "upperBoundary": {"precision": 5, "unit": "Day"},
                    },
                },
                {
                    "name": "Noise Date (Uniform)",
                    "sample_data": "Jul 26, 2022",
                    "n_samples": int(1e5),
                    "data_type": "Date",
                    "method": "NoiseDate",
                    "method_opts": {
                        "distribution": "Uniform",
                        "lowerBoundary": {"precision": 0, "unit": "Day"},
                        "upperBoundary": {"precision": 5, "unit": "Day"},
                    },
                },
            ]
        )
