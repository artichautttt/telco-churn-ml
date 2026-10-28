PY ?= python

.PHONY: all bq test lint

all:            ## toute l'analyse (seeds fixes)
	$(PY) scripts/run_all.py

bq:             ## EDA SQL sur BigQuery (acces GCP + ADC requis) ; BQ_PROJECT=mon-projet
	$(PY) -m src.bq_eda

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests scripts
