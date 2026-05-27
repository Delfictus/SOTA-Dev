# Aleniglipron Expanded Variant Runbook

This runbook queues true perturbed PRISM trajectories. It does not claim
that queued variants have observed trajectory data until the referenced
`trajectory_output_dir` contains PRISM engine outputs.

- Variant count: 1464
- Output root: /home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_b_chronological/expanded_variant_run
- Full observatory target: 80 replicas / 1600 streams

## Execution Order

1. Materialize each `variant_topology_path` from the validated WT holo topology.
2. Run P0 batches first: PGx sentinels, observed-grid extensions, avoid sentinels, and primary true trajectories.
3. Promote to P1/P2/P3 only after P0 trajectory extraction produces signal grid, shear, hysteresis, and pathway outputs.
4. Never relabel `QUEUED_TRUE_PERTURBED_TRAJECTORY` rows as `L5_OBSERVED` without raw PRISM output artifacts.

## Claim Boundary

Computational target/avoid triage only. No biological efficacy claim is made.
