# -*- coding: utf-8 -*-
import subprocess
import unittest

import pandas as pd

from cosmian_anonymization import anonymize_dataframe


class TestAnonymizeDataframe(unittest.TestCase):
    def test_simple_df(self) -> None:
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


class TestAnonymizeCLI(unittest.TestCase):
    def test_simple_cli(self) -> None:
        res = subprocess.run(
            [
                "cosmian-anonymize",
                "./tests/data/sample_data.csv",
                "./tests/data/sample_config.json",
                "./tests/data/out.csv",
            ],
            capture_output=True,
        )

        self.assertEqual(res.returncode, 0)

        df_out = pd.read_csv("tests/data/out.csv", sep=";")
        self.assertEqual(len(df_out.columns), 6)
        self.assertEqual(len(df_out.values), 100)
