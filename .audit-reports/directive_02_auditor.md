# Directive 02 Enforcement Auditor

Audited committed state: `debt-resolution-clean` at `1ab253e0`.

Verdict: PASS. D02 is VERIFIED_RUNTIME.

Clean archive gates from `/tmp/codex_d02_audit_1ab253e0`:

```bash
PYTHONPATH=src python3 -m pytest --collect-only 2>&1 | grep -c "ERROR" || true
# 0

PYTHONPATH=src python3 -m pytest -q --tb=line
# 937 passed, 55 skipped, 8 warnings in 20.03s
```

Source and config evidence:

- `pytest.ini:2` excludes only `.git`, `.rebuild-reference`, and `tools/pocketminer`; `tools/pocketminer` has `0` tracked files.
- Benchmark prototype scripts guard both required benchmark JSON files before import-time reads:
  - `scripts/test_contact_reorg.py:20`
  - `scripts/test_dccm.py:7`
  - `scripts/test_four_stage_decision.py:12`
  - `scripts/test_probe_panel.py:17`
- FEP and preprocessing fixtures skip when untracked fixture files are absent:
  - `tests/test_fep/conftest.py:29`
  - `tests/test_preprocessing/conftest.py:14`
- Track-A calibration artifact tests skip cleanly when artifacts are absent:
  - `tests/test_chem_bald_loop.py:35`
  - `tests/test_gflownet_tb_loss.py:24`
- WRN strict validation guards external artifacts and removes stale `/tmp` report first:
  - `scripts/tests/test_wrn_1522_validation.py:37`
- `PRISM_ROOT` is derived from file location, and `__main__` delegates to pytest:
  - `scripts/test_target_config.py:20`
  - `scripts/test_target_config.py:100`

No broad test masking found.
