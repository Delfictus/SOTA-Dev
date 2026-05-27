# Release Data Manifest

This ledger records external data artifacts required for full production mode but not committed to the source release. Buyer/reviewer smoke tests must not require these files; tests that exercise these paths either create fixtures or skip with an explicit data-dependency message.

## E025-R0.5 Classified External Artifacts

| Logical name | Expected path | Committed? | Stored in R2? | Checksum | Size bytes | Regeneration script | Minimum smoke test? | Full production campaign? |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| GLP1R WT receptor trajectory PDB | `campaigns/glp1r_aleniglipron/phase_2c_de_novo_capture/single_stream_representative/glp1r_6XOX_WT/glp1r_6XOX_WT.ensemble_trajectory.pdb` | no | to verify | `sha256:91623d06eeecc78ca665e69b2734a93a648c2313946cded36acd5ae22836adbe` | 496702139 | `scripts/prism-validate-and-run.sh` de novo capture pipeline | no | yes |
| Scaffold consensus survivor action corpus | `campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_scaffold_consensus_action_corpus.parquet` | no | to verify | `sha256:a7a26e1e7688f67d0823069b5dff506c02c46ff9f8d441dda9a1e4bed18c264c` | 66049338 | `scripts/build_scaffold_consensus_survivor_corpus.py` | no | yes |
| Real512 O3A survivor Z-matrix corpus | `campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_real512_o3a_zmatrix.parquet` | no | to verify | `sha256:0631849cd3bb380352aa2c2c1ad120177c2732ffe08432b2cb2a2183ab63657b` | 364271 | Track A survivor corpus generation pipeline | no | yes |

## E025-R1.0 Machine-Checked Data Classes

The hermetic scanner accepts these glob classes as external data dependencies or generated production outputs. Individual findings remain enumerated in `HERMETIC_INTEGRITY_LEDGER.json`; this table classifies the dependency family without committing private/local artifacts.

| Data class | Manifest pattern | Committed? | Release action |
| --- | --- | --- | --- |
| Track A survivor corpora | `vspace_survivors*.parquet` | mixed | External/generated corpus; restore from release data archive or regenerate. |
| Signal grids | `signal_grid_*.parquet` | mixed | External/generated field data; restore from release data archive or regenerate. |
| Thermodynamic tensors | `*_tensor.parquet` | mixed | External/generated field data; restore from release data archive or regenerate. |
| Differential/pathway fields | `*_differential.parquet` | mixed | External/generated field data; restore from release data archive or regenerate. |
| Translation pathway nodes | `translation_pathway_nodes.parquet` | mixed | External/generated field data; restore from release data archive or regenerate. |
| Ligand conformer SDFs | `*.sdf` | no | External chemistry input; restore from release data archive or regenerate. |
| Ligand conformer directories | `conformers/*.sdf` | no | External chemistry input; restore from release data archive or regenerate. |
| Synthon/anchor tables | `enamine_*.parquet` | mixed | External chemistry corpus; restore from release data archive or regenerate. |
| GLP1R PGx source tables | `GLP1R_*.csv` | mixed | External/source PGx data; restore from release data archive. |
| gnomAD source tables | `gnomAD_*.csv` | mixed | External/source PGx data; restore from release data archive. |
| OpenFF NAGL model | `openff-gnn-am1bcc-1.0.0.pt` | no | External chemistry model dependency. |
| Topology bundles | `*.topology.json` | mixed | Generated topology artifacts; committed fixtures where needed, otherwise restore/regenerate. |
| Residue maps | `*.residue_map.json` | mixed | Generated topology sidecars; committed fixtures where needed, otherwise restore/regenerate. |
| Binding site outputs | `*.binding_sites.json` | mixed | Generated receptor/site outputs; restore/regenerate for full campaigns. |
| KCC visualization outputs | `*.kcc_visualization.json` | mixed | Generated receptor/site outputs; restore/regenerate for full campaigns. |
| Campaign Parquet outputs | `campaigns/**/*.parquet` | mixed | Generated campaign data or external campaign artifact. |
| Campaign JSON outputs | `campaigns/**/*.json` | mixed | Generated campaign data or external campaign artifact. |
| Campaign CSV outputs | `campaigns/**/*.csv` | mixed | Generated campaign data or external campaign artifact. |
| Temporary PRISM prep outputs | `/tmp/prism_prep_*/*.pdb` | no | Generated preprocessing scratch; not required for smoke tests. |
| PRISM scratch outputs | `/mnt/storage/**` | no | Developer/full-production scratch data; configure via `PRISM_SCRATCH_ROOT` or restore from archive. |
| Generic generated Parquet artifacts | `*.parquet` | mixed | Generated or external table artifacts; committed only when promoted to fixture. |
| Generic generated JSON artifacts | `*.json` | mixed | Generated manifests/reports/config sidecars; committed only when source-of-truth fixtures. |
| Generic generated CSV artifacts | `*.csv` | mixed | Generated tabular reports or external source tables; committed only when source-of-truth fixtures. |
| Generic generated YAML artifacts | `*.yaml` | mixed | Generated/config manifests; committed only when source-of-truth fixtures. |
| Generic generated YML artifacts | `*.yml` | mixed | Generated/config manifests; committed only when source-of-truth fixtures. |
| Generic generated PDB artifacts | `*.pdb` | mixed | Generated or external structure files; committed only when small smoke fixtures. |
| Generic generated MOL2 artifacts | `*.mol2` | no | Generated chemistry intermediates; restore/regenerate for full chemistry workflows. |
| Generic generated SDF artifacts | `*.sdf` | no | Generated chemistry intermediates or external ligand structures. |
| Generic generated model artifacts | `*.pt` | no | External/generated ML model checkpoint. |
| Generic generated NumPy artifacts | `*.npy` | no | External/generated numeric arrays. |
| Generic generated compressed NumPy artifacts | `*.npz` | no | External/generated training arrays, tokenizers, and latent stores. |
| Generic generated SQLite artifacts | `*.sqlite` | no | External/generated ontology or audit indexes. |
| Generic R2 artifact references | `r2:*` | no | Cloud archive references; require configured R2 credentials. |
| Environment-expanded artifact references | `$*` | no | Runtime-configured generated paths; documented through environment variables. |

## Packaging Rule

Do not delete or weaken runtime dependencies to make the release hermetic. External production artifacts stay available in developer/full-production mode through documented paths, environment configuration, or archive restoration. Clean release and buyer/reviewer modes must run committed smoke fixtures without relying on these private local files.
