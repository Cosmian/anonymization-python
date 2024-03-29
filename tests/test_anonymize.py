# -*- coding: utf-8 -*-
import subprocess
import unittest

import pandas as pd

from cosmian_anonymization import anonymize_dataframe


class TestAnonymizeDataframe(unittest.TestCase):
    def test_hash_df(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Jane", "Bob", "John"],
                "lastname": ["Smith", "Lemon", "Doe"],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "firstname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "Hash",
                    "methodOptions": {
                        "hashType": "Argon2",
                        "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
                    },
                },
                {
                    "key": "1",
                    "name": "lastname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "Hash",
                    "methodOptions": {"hashType": "SHA3"},
                },
            ],
        }

        df_out = anonymize_dataframe(df, config, inplace=True)
        self.assertEqual(len(df_out.columns), 2)
        self.assertEqual(len(df_out.values), 3)

    def test_fpe_anonymization(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Jane1", "Bob2", "John3"],
                "lastname": ["Smith", "Lemon", "Donald"],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "firstname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "FpeString",
                    "methodOptions": {"alphabet": "alpha", "extendWith": "1234"},
                },
                {
                    "key": "1",
                    "name": "lastname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "FpeString",
                    "methodOptions": {"alphabet": "alpha"},
                    "result": "qvMKSa IDNfs",
                },
            ],
        }

        df_out = anonymize_dataframe(df, config, inplace=True)
        self.assertEqual(len(df_out.columns), 2)
        self.assertEqual(len(df_out.values), 3)

    def test_error_missing_col(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Jane", "Bob", "John"],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "firstname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "Hash",
                    "methodOptions": {
                        "hashType": "Argon2",
                        "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
                    },
                },
                {
                    "key": "1",
                    "name": "lastname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "Hash",
                    "methodOptions": {"hashType": "SHA3"},
                },
            ],
        }
        with self.assertRaises(ValueError):
            anonymize_dataframe(df, config, inplace=True)

    def test_error_type_conversion(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Jane", "Bob", "John"],
                "lastname": ["Smith", "Lemon", "Doe"],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "firstname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "Hash",
                    "methodOptions": {
                        "hashType": "Argon2",
                        "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
                    },
                },
                {
                    "key": "1",
                    "name": "lastname",
                    "type": "Integer",
                    "example": "1",
                    "method": "AggregationInteger",
                    "methodOptions": {"powerOfTen": 2},
                },
            ],
        }

        with self.assertRaises(ValueError):
            anonymize_dataframe(df, config, inplace=True)

    def test_error_anonymization(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Jane", "Bob", "John"],
                "lastname": ["Smith", "Lemon", "Doe"],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "firstname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "Hash",
                    "methodOptions": {
                        "hashType": "Argon2",
                        "saltValue": "53c50914-fe44-4c10-909c-042f49b3ecb0",
                    },
                },
                {
                    "key": "1",
                    "name": "lastname",
                    "type": "Text",
                    "example": "Kenyon",
                    "method": "FpeString",
                    "methodOptions": {"alphabet": "alpha"},
                    "result": "qvMKSa IDNfs",
                },
            ],
        }
        with self.assertRaises(ValueError):
            anonymize_dataframe(df, config, inplace=True)

    def test_working_float_nan(self) -> None:
        df = pd.DataFrame(
            {
                "amount": [1, 2, float("nan")],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "amount",
                    "type": "Float",
                    "example": "1",
                    "method": "RescalingFloat",
                    "methodOptions": {
                        "mean": 2,
                        "stdDev": 1,
                        "scale": 10,
                        "translation": 100,
                    },
                },
            ]
        }

        df_out = anonymize_dataframe(df, config)
        self.assertEqual(len(df_out.columns), 1)
        self.assertEqual(len(df_out.values), 3)

    def test_error_int_nan(self) -> None:
        df = pd.DataFrame(
            {
                "amount": [1, 2, float("nan")],
                "duration": [3, 4, float("-inf")],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "amount",
                    "type": "Integer",
                    "example": "1",
                    "method": "RescalingInteger",
                    "methodOptions": {
                        "mean": 2,
                        "stdDev": 1,
                        "scale": 10,
                        "translation": 100,
                    },
                },
            ]
        }

        with self.assertRaises(ValueError):
            anonymize_dataframe(df, config)

    def test_no_method(self) -> None:
        df = pd.DataFrame(
            {
                "lastname": ["Smith", "Lemon", "Doe"],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "lastname",
                    "type": "Text",
                    "example": "Smith",
                },
            ],
        }

        df_out = anonymize_dataframe(df, config, inplace=True)
        self.assertEqual(len(df_out.columns), 1)
        self.assertEqual(len(df_out.values), 3)

    def test_delete_col(self) -> None:
        df = pd.DataFrame(
            {
                "lastname": ["Smith", "Lemon", "Doe"],
            }
        )

        config = {
            "metadata": [
                {
                    "key": "0",
                    "name": "lastname",
                    "type": "Text",
                    "example": "Smith",
                    "method": "DeleteColumn",
                },
            ],
        }

        df_out = anonymize_dataframe(df, config, inplace=True)
        self.assertEqual(len(df_out.columns), 0)


class TestAnonymizeCLI(unittest.TestCase):
    def test_simple_cli(self) -> None:
        res = subprocess.run(
            [
                "cosmian-anonymize",
                "./tests/data/sample_data.csv",
                "./tests/data/sample_config.json",
                "./tests/data/out_sample.csv",
            ],
            capture_output=True,
        )

        self.assertEqual(res.returncode, 0)

        df_out = pd.read_csv("tests/data/out_sample.csv", sep=";")
        self.assertEqual(len(df_out.columns), 6)
        self.assertEqual(len(df_out.values), 100)

    def test_demo_cli(self) -> None:
        res = subprocess.run(
            [
                "cosmian-anonymize",
                "./tests/data/data_demo.csv",
                "./tests/data/config_demo.json",
                "./tests/data/out_demo.csv",
            ],
            capture_output=True,
        )

        self.assertEqual(res.returncode, 0)

        df_out = pd.read_csv("tests/data/out_demo.csv", sep=";")
        self.assertEqual(len(df_out.columns), 5)
        self.assertEqual(len(df_out.values), 100)
