# BLIND VALIDATION AUDIT SCHEMA
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Purpose

Defines the required audit trail fields for every artifact in the blind validation. Enables independent reproduction and reviewability.

---

## Per-run audit record (binding_sites.json header)

Required fields in every PRISM4D output:
```json
{
  "run_metadata": {
    "target": "HRAS_Q61H",
    "slot": "B01",
    "apo_pdb": "4L9S",
    "chain": "A",
    "engine_commit": "f8f368f6",
    "run_timestamp_utc": "2026-05-13T...",
    "replica_seed": 42,
    "spike_percentile": 50,
    "multi_stream": 8,
    "flags": "--fast --hysteresis --prism-therm ...",
    "frozen": false
  }
}
```

If the engine does not write run_metadata, record it in a companion `run_metadata.json` in the run dir.

---

## Freeze audit record (FREEZE_METADATA_*.txt)

```
FREEZE_TIMESTAMP: 2026-05-13T...Z
TARGET: B01_HRAS_Q61H
OPERATOR: diddy
ENGINE_COMMIT: f8f368f6
HOLO_ACCESS: NONE_BEFORE_FREEZE
MANIFEST: FREEZE_MANIFEST_B01_HRAS_Q61H.sha256
```

---

## Post-freeze validator audit record

prism_pub_baseline_validator.py writes `validation_report.md` which must contain:
- Target name
- Apo PDB used
- Holo references downloaded + aligned
- Alignment RMSD and seqid per holo
- Shell distances per site per holo
- SR@k values
- LORO results
- Run timestamp

---

## Provenance chain (forensic)

```
Raw PDB (RCSB) 
  → prism-clean.py (strip altconfs, validate diversity)
  → prism-prep (topology + NMA)
  → prism-validate-and-run.sh (engine; SHA256 of input topology)
  → binding_sites.json + kcc_visualization.json (frozen artifacts)
  → FREEZE_MANIFEST (SHA256 of all frozen artifacts)
  → git commit (timestamp in history)
  → prism_pub_baseline_validator.py (post-freeze only)
  → BLIND_VALIDATION_FINAL_REPORT.md
```

Each arrow is a traceable step with timestamp, input hash, and output.

---

## SHA256 audit commands

```bash
# Verify input topology hash (record at freeze time)
sha256sum ${BLIND_BASE}/${TARGET}/topologies/*.topology.json

# Verify frozen artifacts
cd ${BLIND_BASE}/${TARGET}/frozen
sha256sum -c FREEZE_MANIFEST_${TARGET}.sha256

# Verify post-freeze scoring inputs match frozen
diff \
  <(sha256sum ${BLIND_BASE}/${TARGET}/run/*.binding_sites.json | awk '{print $1}') \
  <(grep "binding_sites.json" ${BLIND_BASE}/${TARGET}/frozen/FREEZE_MANIFEST_*.sha256 | awk '{print $1}')
```

---

## Version pinning

| Component | Version | Verification command |
|-----------|---------|---------------------|
| PRISM4D engine | f8f368f6 | git rev-parse HEAD |
| fpocket | 4.2.3 | fpocket --version |
| P2Rank | 2.4.2 | /opt/p2rank/prank --version |
| prism_pub_baseline_validator.py | git HEAD version | git log -1 scripts/quarantine/prism_pub_baseline_validator.py |
