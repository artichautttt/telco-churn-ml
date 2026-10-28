"""Configuration pytest commune.

Marqueur `bigquery` : test qui appelle reellement BigQuery (identifiants
GCP requis). Ignore en CI (variable CI definie par GitHub Actions) et quand
SKIP_BIGQUERY=1. Les tests actuels de src.bq_eda utilisent un faux client
et ne portent pas ce marqueur.
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "bigquery: necessite un acces BigQuery reel")


def pytest_collection_modifyitems(config, items):
    if not (os.environ.get("CI") or os.environ.get("SKIP_BIGQUERY") == "1"):
        return
    skip = pytest.mark.skip(reason="pas d'acces BigQuery en CI")
    for item in items:
        if "bigquery" in item.keywords:
            item.add_marker(skip)
