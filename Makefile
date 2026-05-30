PYTHON ?= python3
MYPY ?= $(PYTHON) -m mypy

.PHONY: gate-phase0 release-acceptance release-preseal-agents release-postseal-agents release-gate

gate-phase0:
	$(PYTHON) scripts/ci/ban_check.py
	$(PYTHON) scripts/ci/parquet_provenance_check.py
	$(MYPY) --strict src/prism_dstw

release-acceptance:
	@if [ -z "$$PRISM_RELEASE_ROOT" ]; then echo "PRISM_RELEASE_ROOT is required"; exit 1; fi
	$(PYTHON) scripts/verify_restored_release.py --release-root "$$PRISM_RELEASE_ROOT" --run-smoke --run-db-checks --run-cuda-checks --run-candidate-checks

release-preseal-agents:
	@if [ -z "$$PRISM_INPUT_ROOT" ]; then echo "PRISM_INPUT_ROOT is required"; exit 1; fi
	@if [ -z "$$PRISM_REPO_ROOT" ]; then echo "PRISM_REPO_ROOT is required"; exit 1; fi
	@if [ -z "$$PRISM_RELEASE_ROOT" ]; then echo "PRISM_RELEASE_ROOT is required"; exit 1; fi
	$(PYTHON) scripts/run_release_verification_agents.py --input-root "$$PRISM_INPUT_ROOT" --repo-root "$$PRISM_REPO_ROOT" --release-root "$$PRISM_RELEASE_ROOT" --phase pre-seal --parallel --fail-closed

release-postseal-agents:
	@if [ -z "$$PRISM_INPUT_ROOT" ]; then echo "PRISM_INPUT_ROOT is required"; exit 1; fi
	@if [ -z "$$PRISM_REPO_ROOT" ]; then echo "PRISM_REPO_ROOT is required"; exit 1; fi
	@if [ -z "$$PRISM_RELEASE_ROOT" ]; then echo "PRISM_RELEASE_ROOT is required"; exit 1; fi
	$(PYTHON) scripts/run_release_verification_agents.py --input-root "$$PRISM_INPUT_ROOT" --repo-root "$$PRISM_REPO_ROOT" --release-root "$$PRISM_RELEASE_ROOT" --phase post-seal --parallel --fail-closed

release-gate: release-preseal-agents release-postseal-agents
