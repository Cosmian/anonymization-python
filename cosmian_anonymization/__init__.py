# -*- coding: utf-8 -*-
import importlib.metadata

from cosmian_anonymization.anonymize import anonymize_dataframe, anonymize_from_files

__version__ = importlib.metadata.version(__package__ or __name__)
__all__ = ["anonymize_dataframe", "anonymize_from_files"]
