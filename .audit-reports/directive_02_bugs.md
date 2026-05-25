# Directive 02 Bug Hunter

Audited committed state: `debt-resolution-clean` at `1ab253e0`.

Verdict: no CRITICAL or HIGH bugs remain.

Validation:

- Clean `git archive 1ab253e0` checkout collected `988` tests with exit `0`.
- Clean archive full root pytest: `937 passed, 55 skipped, 8 warnings in 20.28s`.
- No broad masking found: `pytest.ini` only excludes `.git`, `.rebuild-reference`, and `tools/pocketminer`.
- Direct script execution: `python3 scripts/test_target_config.py` exits `0`.
- Prior clean-checkout HIGH is resolved: missing local fixtures now skip instead of failing.

Findings:

## BUG_1

severity: MEDIUM

location: D02 literal gate command

description: The exact directive pipeline `PYTHONPATH=src python3 -m pytest --collect-only 2>&1 | grep -c "ERROR"` prints `0` on clean collection but exits `1` because `grep` returns nonzero when there are no matches.

impact: Any `set -e` caller using the directive command literally can falsely fail D02 despite zero collection errors.

status: documented. The committed `scripts/regression_gate.sh` uses the safe form with `|| true`, and all D02 audits record the count with `|| true`.

## BUG_2

severity: LOW

location: `scripts/test_contact_reorg.py`, `scripts/test_dccm.py`, `scripts/test_four_stage_decision.py`, `scripts/test_probe_panel.py`

description: Targeted pytest runs of only these module-skipped prototype scripts still return exit code `5` because no tests are collected. Root pytest is fine and records them as skips.

impact: Root suite is stable, but narrow CI shards targeting only these files could fail.

status: documented as low severity. No CRITICAL/HIGH blocker remains for D02 progression.
