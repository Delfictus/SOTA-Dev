# PANEL OUTPUT ROOT REDIRECT — 2026-04-18

## Old root (retained for archival + R2 offload)
```
/mnt/storage/prism-outputs/twin-10-patent/
/mnt/storage/prism-outputs/m1-strict-dcc-panel/
```

## New root (for future panel runs)
```
~/prism-working/twin-10-patent/
~/prism-working/m1-strict-dcc-panel/
```

## Rationale
`/mnt/storage` hit 100% ENOSPC during the strict_dcc_panel_v1 run because
completed-target raw data was being preserved pending R2 verification.
Incremental deletion has recovered >150 GB and the mount is now at 92%,
but future panel runs should write to a local working directory that does
not compete with R2-offload-pending completed data.

## Active uploader status
PID 2008059 (R2 large-phase upload) is still running and reading from the
old root. Do NOT move or rename `/mnt/storage/prism-outputs/...` while the
uploader is active. New panel runs use the new root without disturbing the
uploader's read path.

## Scripts that reference the old root
```
scripts/quarantine/build_m1_panel.py      (PANEL_BASE + target_dir literals)
scripts/quarantine/b_no_therm_feature_gap.py
scripts/quarantine/canonical_dcc_audit.py
scripts/quarantine/classifier_audit.py
scripts/quarantine/classifier_audit_expanded.py
scripts/quarantine/m1_ablation.py
scripts/quarantine/m1_1_manifest.py
scripts/quarantine/per_target_completion_block.py
scripts/quarantine/spike_metadata_inventory.py
scripts/quarantine/engine_full_harvest.py
scripts/quarantine/site_vs_holo_strict.py
scripts/quarantine/completeness_certifier.py
scripts/quarantine/r2_offload_completed_targets.py
```

**These scripts read from the OLD root** (reading completed-target data that
is currently either local or uploaded to R2). Do NOT change their read
paths — they point at the correct archival data.

## Scripts that WRITE new panel output

Only `build_m1_panel.py` WRITES to the panel output root (target_configs +
launcher). It is the single point that must be redirected for future panels.

## Single-line redirect

Adjust `build_m1_panel.py` before launching any new panel:

```python
# OLD
PANEL_BASE = Path("/mnt/storage/prism-outputs/m1-strict-dcc-panel")

# NEW
PANEL_BASE = Path.home() / "prism-working" / "m1-strict-dcc-panel"
```

And adjust `target_dir` fields for TWIN-10 recovery entries similarly.
