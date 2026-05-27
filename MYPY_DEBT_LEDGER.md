# Mypy Strict Debt Ledger

Command:

```bash
PYTHONPATH=src python3 -m mypy --strict src/ scripts/
```

Result: FAIL, with `4852` errors in `301` files, checked across `553` source files.

This ledger is the explicit E025-R1.0 gate artifact for the allowed path where `mypy --strict` does not yet pass. The failure is broad legacy typing debt in scripts and research tooling; no runtime functionality was removed or stubbed.

## Error Categories

| Count | Mypy code |
| ---: | --- |
| 1603 | `no-untyped-call` |
| 1285 | `no-untyped-def` |
| 508 | `type-arg` |
| 188 | `var-annotated` |
| 176 | `assignment` |
| 150 | `index` |
| 134 | `arg-type` |
| 128 | `attr-defined` |
| 123 | `import-untyped` |
| 122 | `no-any-return` |
| 112 | `operator` |
| 74 | `union-attr` |
| 71 | `import-not-found` |
| 40 | `call-overload` |
| 22 | `return-value` |
| 22 | `str-bytes-safe` |
| 21 | `misc` |
| 20 | `dict-item` |
| 15 | `unused-ignore` |
| 13 | `name-defined` |

## Highest-Debt Files

| Count | File |
| ---: | --- |
| 111 | `scripts/dossier_unified.py` |
| 92 | `scripts/run_bench10.py` |
| 89 | `scripts/egnn_pocket_ranker_v2.py` |
| 87 | `scripts/run_hard_targets.py` |
| 74 | `scripts/full_spectrum_extraction.py` |
| 71 | `scripts/dossier_full.py` |
| 71 | `scripts/quarantine/prism_pub_baseline_validator.py` |
| 64 | `scripts/benchmark_comparison.py` |
| 62 | `scripts/quarantine/prism_manifold_shell_validator.py` |
| 60 | `scripts/production/validate_v4_contract.py` |
| 58 | `scripts/quarantine/gate_a_validator.py` |
| 58 | `scripts/phase_manifold_ranker.py` |
| 54 | `scripts/validate_kras_residue_overlap.py` |
| 54 | `scripts/quarantine/rebuild_patent_docx_v2.py` |
| 51 | `scripts/quarantine/build_true_apo_manifest.py` |
| 50 | `scripts/interfaces/site_spike_view.py` |
| 49 | `scripts/prism-corpus-status.py` |
| 48 | `scripts/quarantine/pfr_render_panels.py` |
| 48 | `scripts/training/prism_only_feature_extractor_v5.py` |
| 47 | `scripts/train_hysteresis_predictor.py` |

## Remediation Plan

1. Add strict typing first to shared `src/prism_dstw` APIs and production entrypoints.
2. Move quarantine and one-off research scripts behind a separate mypy target before enforcing strict globally.
3. Add or vendor stubs for scientific/visualization libraries that lack `py.typed`.
4. Track error-count burn-down per file; do not weaken E025 release packaging to hide typing debt.
