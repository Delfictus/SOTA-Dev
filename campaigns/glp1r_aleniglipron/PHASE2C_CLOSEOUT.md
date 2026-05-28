# Phase 2C Closeout

Status: complete and archived.

## Local Seal

- Sealed manifest: `campaigns/glp1r_aleniglipron/PHASE2C_SEALED_MANIFEST.json`
- Schema: `phase2c_sealed_manifest_v1`
- Replicas: `16 / 16`
- Total MD spikes: `1204768024`
- Record spike sum: `1204768024`
- Closeout gate: `PASS`

All 16 records have:

- `completion_status=md_evidence_complete_postmd_deferred`
- `validation_status=accepted_required_artifacts_present`
- `required_artifacts_complete=true`
- `streams_completed=20`
- `stream_count=20`
- `streams_serialized=20`
- `schema_kind=md_evidence_manifest`
- `binding_sites_status=not_materialized_md_only`

## R2 Archive

- Source: `campaigns/glp1r_aleniglipron/phase_2c_snapshots`
- Destination: `r2:prism-archive/glp1r_aleniglipron/phase_2c_snapshots`
- Payload copied: `147.831 GiB`
- Files copied: `2992 / 2992`
- Copy exit: `0`
- Check result: `0 differences found`
- Matching files: `2992`
- Check exit: `0`

Archive logs:

- `phase2c_r2_archive_20260527_230652.log`
- `phase2c_r2_check_20260527_231615.log`

## Boundary

Phase 2C is an MD-only evidence seal. Post-MD binding-site materialization is intentionally deferred:

- No raw Phase 2C mutation.
- No hidden `forces_final` to `asc_vectors` alias.
- No dynamic voxel exporter as primary Phase 2C completion path.
- Downstream materialization must consume the sealed `md_evidence_manifest.json` records.
