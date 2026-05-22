PYTHON ?= python3
MYPY ?= $(PYTHON) -m mypy

.PHONY: gate-phase0

gate-phase0:
	$(PYTHON) scripts/ci/ban_check.py
	$(PYTHON) scripts/ci/parquet_provenance_check.py
	$(MYPY) --strict src/prism_dstw
