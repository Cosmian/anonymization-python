# anonymization-python

[![PyPI version](https://badge.fury.io/py/cosmian-anonymization.svg)](https://badge.fury.io/py/cosmian-anonymization)
![Build status](https://github.com/Cosmian/anonymization-python/actions/workflows/ci.yml/badge.svg)

Cosmian Anonymization library in Python

## Usage

- As a library

```python
from cosmian_anonymization import anonymize_dataframe

try:
    # Anonymize the raw data according to a configuration.
    anonymized_df = anonymize_dataframe(df_raw, config)
except ValueError as e:
    print("Anonymization failed:", e)
```

- From command line

```bash
cosmian-anonymize <input_csv> <input_config> <output_csv>
```

## Build from source

- Install dependencies

```bash
poetry install
```

- Build package

```bash
poetry build
```

- Run tests

```bash
poetry run python -m unittest tests/test*.py
```
