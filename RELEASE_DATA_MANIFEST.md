# Release Data Manifest

This ledger records external data artifacts required for full production mode but not committed to the source release. Buyer/reviewer smoke tests must not require these files; tests that exercise these paths either create fixtures or skip with an explicit data-dependency message.

## E025-R0.5 Classified External Artifacts

| Logical name | Expected path | Committed? | Stored in R2? | Checksum | Size bytes | Regeneration script | Minimum smoke test? | Full production campaign? |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| GLP1R WT receptor trajectory PDB | `campaigns/glp1r_aleniglipron/phase_2c_de_novo_capture/single_stream_representative/glp1r_6XOX_WT/glp1r_6XOX_WT.ensemble_trajectory.pdb` | no | to verify | `sha256:91623d06eeecc78ca665e69b2734a93a648c2313946cded36acd5ae22836adbe` | 496702139 | `scripts/prism-validate-and-run.sh` de novo capture pipeline | no | yes |
| Scaffold consensus survivor action corpus | `campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_scaffold_consensus_action_corpus.parquet` | no | to verify | `sha256:a7a26e1e7688f67d0823069b5dff506c02c46ff9f8d441dda9a1e4bed18c264c` | 66049338 | `scripts/build_scaffold_consensus_survivor_corpus.py` | no | yes |
| Real512 O3A survivor Z-matrix corpus | `campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_real512_o3a_zmatrix.parquet` | no | to verify | `sha256:0631849cd3bb380352aa2c2c1ad120177c2732ffe08432b2cb2a2183ab63657b` | 364271 | Track A survivor corpus generation pipeline | no | yes |

## Packaging Rule

Do not delete or weaken runtime dependencies to make the release hermetic. External production artifacts stay available in developer/full-production mode through documented paths, environment configuration, or archive restoration. Clean release and buyer/reviewer modes must run committed smoke fixtures without relying on these private local files.
