# Deliverable A: Receptor Durability Audit V0.1

Generated: 2026-05-22T17:37:02+00:00

## Executive Summary

This receptor-side durability audit summarizes the 80-replica unliganded/WT thermodynamic manifold across 1600 production streams. The final risk ranking is derived from the fused seven-tensor receptor durability map and synchronized into the Track 0 analyst workbook.

**Claim Boundaries:** This audit provides computationally bounded receptor-side mechanistic evidence. Analog-conditioned interference is provided via expert-in-the-loop emulation (V0.1). This document does not claim in vivo clinical durability.

## The Thermodynamic Manifold

The posthoc manifold contains 17 channel summaries across durability classes and receptor conditions. The mechanical validation pass pruned **437** false-positive edges before Track 0 target selection.

| Condition | Durability Class | Edges | Mean Risk | Max Risk | Masked Spikes | Violent Rupture Edges | Mechanically Pruned |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| glp1r_5VEX_WT | mechanically_pruned | 207 | 0.000000 | 0.000000 | 0.0 | 0 | 207 |
| glp1r_5VEX_WT | moderate_durability_risk | 344 | 1.499182 | 2.021576 | 0.0 | 0 | 0 |
| glp1r_6LN2_A316T | mechanically_pruned | 76 | 0.000000 | 0.000000 | 0.0 | 0 | 76 |
| glp1r_6LN2_A316T | moderate_durability_risk | 547 | 1.575695 | 1.882116 | 0.0 | 0 | 0 |
| glp1r_6LN2_T149M | mechanically_pruned | 75 | 0.000000 | 0.000000 | 0.0 | 0 | 75 |
| glp1r_6LN2_T149M | moderate_durability_risk | 547 | 1.644293 | 1.925600 | 0.0 | 0 | 0 |
| glp1r_6LN2_WT | mechanically_pruned | 79 | 0.000000 | 0.000000 | 0.0 | 0 | 79 |
| glp1r_6LN2_WT | moderate_durability_risk | 538 | 1.582074 | 1.868364 | 0.0 | 0 | 0 |
| glp1r_6X1A_WT | elevated_durability_risk | 197 | 2.186804 | 2.262484 | 0.0 | 0 | 0 |
| glp1r_6X1A_WT | moderate_durability_risk | 424 | 2.069861 | 2.152070 | 0.0 | 0 | 0 |
| glp1r_6XOX_A316T | elevated_durability_risk | 467 | 2.234120 | 2.355776 | 0.0 | 0 | 0 |
| glp1r_6XOX_A316T | moderate_durability_risk | 135 | 2.103695 | 2.151775 | 0.0 | 0 | 0 |
| glp1r_6XOX_T149M | elevated_durability_risk | 62 | 2.171969 | 2.225716 | 0.0 | 0 | 0 |
| glp1r_6XOX_T149M | moderate_durability_risk | 531 | 2.062237 | 2.151583 | 0.0 | 0 | 0 |
| glp1r_6XOX_WT | critical_durability_risk | 243 | 15.727915 | 17.037603 | 184876.0 | 0 | 0 |
| glp1r_6XOX_WT | elevated_durability_risk | 240 | 13.276202 | 14.905069 | 37482.0 | 0 | 0 |
| glp1r_6XOX_WT | moderate_durability_risk | 121 | 1.748853 | 1.840139 | 0.0 | 0 | 0 |

## Critical Durability Risks

Selection policy: critical_durability_risk rows inner-joined to sar_steric_interface_catalog.parquet on condition_id, edge_from_residue, edge_to_residue, and edge_class; edge scores come from the edge-level evaluator, not from node-to-edge broadcasting.

| Rank | Edge ID | Edge Class | Condition | Residue Edge | Contact A | SignedTE | Raw Risk | BOCPD Survival ps | DT Drops | Violent DT Drops | Steering Prior | Variance Penalty | Mechanical Load |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | glp1r_6XOX_WT:LEU245->TYR241:pocket_vector | pocket_vector | glp1r_6XOX_WT | LEU245 -> TYR241 | 4.023 | 0.000867 | 17.037603 | 0.005993 | 0.0 | 0.0 | 1.501156 | 3.320117 | 1263.606043 |
| 2 | glp1r_6XOX_WT:ARG310->TYR241:pocket_vector | pocket_vector | glp1r_6XOX_WT | ARG310 -> TYR241 | 4.618 | 0.004981 | 16.936943 | 0.005993 | 0.0 | 0.0 | 1.507644 | 3.320117 | 1094.717251 |
| 3 | glp1r_6XOX_WT:PHE230->GLN234:pocket_vector | pocket_vector | glp1r_6XOX_WT | PHE230 -> GLN234 | 4.248 | 0.002668 | 16.910155 | 0.005993 | 0.0 | 0.0 | 1.518651 | 3.320117 | 1072.660327 |
| 4 | glp1r_6XOX_WT:ASN320->TYR241:pocket_vector | pocket_vector | glp1r_6XOX_WT | ASN320 -> TYR241 | 4.070 | 0.006483 | 16.782982 | 0.005993 | 0.0 | 0.0 | 1.519812 | 3.320117 | 1253.424587 |
| 5 | glp1r_6XOX_WT:TYR241->ARG310:downstream_lock | downstream_lock | glp1r_6XOX_WT | TYR241 -> ARG310 | 4.618 | -0.004981 | 16.936943 | 0.005993 | 0.0 | 0.0 | 1.507644 | 3.320117 | 1094.717251 |
| 6 | glp1r_6XOX_WT:TYR241->ILE317:downstream_lock | downstream_lock | glp1r_6XOX_WT | TYR241 -> ILE317 | 4.610 | -0.004190 | 16.622596 | 0.005993 | 0.0 | 0.0 | 1.489289 | 3.320117 | 1211.116232 |
| 7 | glp1r_6XOX_WT:TYR241->ALA316:downstream_lock | downstream_lock | glp1r_6XOX_WT | TYR241 -> ALA316 | 4.514 | -0.006502 | 16.566392 | 0.005993 | 0.0 | 0.0 | 1.512557 | 3.320117 | 1158.980611 |
| 8 | glp1r_6XOX_WT:MET303->LEU307:downstream_lock | downstream_lock | glp1r_6XOX_WT | MET303 -> LEU307 | 4.098 | 0.001912 | 16.541481 | 0.005993 | 0.0 | 0.0 | 1.445315 | 3.320117 | 1154.031040 |
| 9 | glp1r_6XOX_WT:TYR242->ALA316:downstream_lock | downstream_lock | glp1r_6XOX_WT | TYR242 -> ALA316 | 4.394 | -0.004950 | 16.433762 | 0.005993 | 0.0 | 0.0 | 1.471814 | 3.320117 | 1153.655107 |

The pocket-vector and downstream-lock cohorts are selected only after the receptor durability map is inner-joined to the SAR steric interface catalog and residue index mapping matrix. Residue names use canonical three-letter amino acid codes plus biological sequence numbers.

## Track 0 Workbook Synchronization

The Track 0 workbook was regenerated with 5 placeholder analogs across 9 dynamically selected receptor edges, producing 45 analyst-scoring rows.

Workbook path: `campaigns/glp1r_aleniglipron/track_0_manual_emulation/Track_0_Interference_Workbook.csv`

## Cryptographic Lineage

| Source Parquet | SHA-256 |
| --- | --- |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/receptor_durability_risk_map.parquet` | `7e5ae4d15679971aa2c4fdf24e3e15e2499eaebed5987f55b10ca32184d7bac2` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/receptor_durability_channel_summary.parquet` | `b64c12c22d133071a3ba726892f4eb879d375201892b8402be3c51782a93af71` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/spike_events_snr_masked.parquet` | `4d132a17a5542233d1b5431356973295a205cb9050a1cad3fe7df5b26e7296fc` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet` | `c8dee22395bbd5dfec96830084e501fef8608280207c4d42fc99a7501022b30c` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/mechanical_load_network.parquet` | `3a85a6fa8e325fe2caf744185de4e19f65b17f9992ef0628e8e3b083f30b1871` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/bocpd_survival_regimes.parquet` | `fc284c42cd659d7c7ec5fd7a8e37efb06bfd9580ad015d766a4f76fb9ceb2afa` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/kinetic_strain_events.parquet` | `14f905112d8ec4045df3917ecff6da8cd99063fdb555338c53c0bb46a450e77e` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/autonomous_steering_tensor.parquet` | `bf13b98b81bf876c564acebb9896004f8893b2eee15d4a9ae5a860802235bddd` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/aromatic_reorganization_tensor.parquet` | `3be6f9b0efe0b399f03477100ca67e9220caf411a0af4c848c766342e305930c` |
| `campaigns/glp1r_aleniglipron/topology/residue_index_mapping_matrix.parquet` | `3bd31a0100f05358125cd918bb62c2c90617b2ed397801a22cc798405f020fd7` |
| `campaigns/glp1r_aleniglipron/integrated_spike_events/full/sar_steric_interface_catalog.parquet` | `dd0c7163d44e67c4f80df065dd8309dd519bcc6372213d987b83765fa340089f` |
