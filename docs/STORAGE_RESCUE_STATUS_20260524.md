# PRISM Storage Rescue Status

Observed: 2026-05-24 22:11:54 PDT
Repository: `/home/diddy/Desktop/Prism4D-bio`
Manifest root: `/mnt/storage/prism-rescue-manifests/Prism4D-bio-20260524T2004PDT`

## Preservation Result

- Source checkpoint pushed earlier: `a6de31dc chore: checkpoint prism storage rescue source state`
- Current pre-status HEAD observed: `06b89cb7 feat(track-a): add tripartite bias scoring and dossiers`
- R2 rescue prefix: `r2:prism-deep-archive-20260516/storage-rescue/Prism4D-bio-20260524T2004PDT`
- Manifest upload verification: 29 matching files, 0 differences, 0 missing on destination
- Priority artifact verification: 4 OK, 0 FAIL
- Secondary runtime artifact upload: 31 `.scratch` files, 707,332,154 bytes
- Secondary runtime verification: 28 stable files matched by R2 download check; 3 unstable files verified by remote SHA-256
- Late-freeze runtime delta: 9 additional `.scratch` files uploaded and verified after repeated training restarts were stopped
- Late PGx screen artifacts: 3 files uploaded and verified after a PGx screening job was stopped
- Cleanup performed: none

## Priority Artifacts Verified

```text
17675f7945ecf6e281aa1d59588bfbba8e1901e22898901bb5c96bfd22f33d21  campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/spike_events_snr_masked.parquet
3e3a6faa153ee426564a3fa7350d1826ea013d664fef04a2047abc2f0885b084  PRISM_GLP1R_M2_Release_v1.0.tar.gz
c3671c4e18baa046e186bfc4584254d507cddcdfa8983f9bf8d5c6e3502807dc  campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards/shard_0001.parquet
c7c599b7404805c5786c8bbb4c72750a8b987603943d34ec16e8c28d09b316f8  campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards/shard_0000.parquet
```

The authoritative R2 verification files are:

- `/mnt/storage/prism-rescue-manifests/Prism4D-bio-20260524T2004PDT/priority_artifacts.remote_verify_noprogress.tsv`
- `/mnt/storage/prism-rescue-manifests/Prism4D-bio-20260524T2004PDT/priority_artifacts.remote_noprogress.sha256`

An earlier verifier attempt using `rclone cat -P` is retained as evidence but is invalid because progress output contaminates stdout in this environment.

## Secondary Runtime Artifacts

Secondary runtime artifacts were uploaded under:

```text
r2:prism-deep-archive-20260516/storage-rescue/Prism4D-bio-20260524T2004PDT/secondary-runtime-artifacts
```

The stable subset is verified by:

- `/mnt/storage/prism-rescue-manifests/Prism4D-bio-20260524T2004PDT/secondary-check-reports-stable/r2_secondary_stable_match.txt`

Three files changed locally after upload because a GFlowNet training wrapper restarted during the rescue. Their already-uploaded remote copies were verified directly against the pre-transfer SHA-256 manifest:

```text
.scratch/epoch016_telemetry.log
.scratch/oracle_batch.parquet
.scratch/oracle_rewards.parquet
```

The authoritative unstable-file verification file is:

- `/mnt/storage/prism-rescue-manifests/Prism4D-bio-20260524T2004PDT/secondary_unstable_remote_verify.tsv`

The restarted GFlowNet trainer and oracle scorer were stopped with SIGTERM, and the follow-up producer scan showed no active `gflownet`, `oracle_scorer`, `vspace`, `ingest`, `am1bcc`, package, or cargo producer.

Additional restart attempts from `pts/15` and a non-tty `bash -lc` wrapper were also stopped by process group. After no `.scratch` writers remained, the final late-freeze runtime delta was uploaded and verified under:

```text
r2:prism-deep-archive-20260516/storage-rescue/Prism4D-bio-20260524T2004PDT/late-freeze-runtime-delta-final2
```

Verification result for that final delta: 9 matching files, 0 differences, 0 missing on destination.

A late PGx screening job was also stopped after it produced a small parquet/report set. Those files were uploaded and verified under:

```text
r2:prism-deep-archive-20260516/storage-rescue/Prism4D-bio-20260524T2004PDT/late-pgx-screen-artifacts
```

Verification result for the late PGx set: 3 matching files, 0 differences, 0 missing on destination.

## Cleanup Gate

Root filesystem is still under pressure at 97% use with about 110 GiB available. The large local artifacts are now hashed, uploaded, and remotely verified, but no local deletion, move, compression, truncation, or cleanup has been performed. Cleanup still requires explicit approval.

Residual local work remains outside this cleanup gate: non-critical generated release/deliverable directories, `.codex/review` files, tracked `__pycache__` modifications, and other non-source artifacts were intentionally not included in the source checkpoint.
