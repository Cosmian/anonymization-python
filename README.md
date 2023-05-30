# anonymization-python

Cosmian Anonymization library in Python

## Usage

- As a library

```python
from cosmian_anonymization import anonymize_dataframe

df_anonymized = anonymize_dataframe(df_raw, config)
```

- Command line interface

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
poetry run python -m unittest tests/*.py
```
