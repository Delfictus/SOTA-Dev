# Runtime Consistency Audit

row_count: 11

| check | status | risk_flags | evidence |
| --- | --- | --- | --- |
| required manifests exist | PASS | [] | ["campaigns/glp1r_aleniglipron/Phase3_PGx_Exclusion_Manifest.json", "campaigns/glp1r_aleniglipron/GLP1R_LIGAND_SET_MANIFEST_v1.parquet", "campaigns/glp1r_aleniglipron/M2_Replayability_Manifest.json", "campaigns/glp1r_... |
| hashes exist | PASS | [] | ["crates/prism-gpu/target/ptx/amber_bonded.ptx.sha256", "crates/prism-gpu/target/ptx/ultimate_md.ptx.sha256", "crates/prism-gpu/target/ptx/dendritic_whcr.ptx.sha256", "crates/prism-gpu/target/ptx/tda.ptx.sha256", "cra... |
| statuses valid | PASS | [] | ["campaigns/glp1r_aleniglipron/phase_2c_de_novo_capture/glp1r_6XOX_WT/ghost_lattice_routing_status.json", "campaigns/glp1r_aleniglipron/track_a_generative/autonomous_infra_status_epoch017.json", "campaigns/glp1r_aleni... |
| no CPU fallback in production captured-tile run | PASS | [] | ["crates/prism-physics/src/cma_cpu.rs"] |
| no uncaptured tile fallback | MISSING | [] | [] |
| reward source canonical | PASS | [] | ["campaigns/glp1r_aleniglipron/track_0_manual_emulation/pubchem_cid_164809721_canonical_smiles.json", "campaigns/glp1r_aleniglipron/track_a_generative/reward_progression.csv", "campaigns/glp1r_aleniglipron/track_a_gen... |
| C6.7 verdict distribution present | MISSING | [] | [] |
| tile credit exists | MISSING | [] | [] |
| cache diagnostics present | PASS | [] | ["src/prism_dstw/orchestration/__pycache__/topology_compiler.cpython-312.pyc", "src/prism_dstw/orchestration/__pycache__/campaign_dispatcher.cpython-312.pyc", "src/prism_dstw/orchestration/__pycache__/reward_function.... |
| operator owner consistent | PASS | [] | ["campaigns/glp1r_aleniglipron/track_b_chronological/subtb_spectral/captured_graph_tiles/restricted_c6_operator_state.json", "src/prism_dstw/gflownet/bsr_operator_state.py", "src/prism_dstw/gflownet/tile_operator_delt... |
| exact command provenance exists | PASS | [] | ["campaigns/glp1r_aleniglipron/provenance/airgap_handoff_manifest.json", "campaigns/glp1r_aleniglipron/provenance/rust_python_schema_audit.md", "campaigns/glp1r_aleniglipron/provenance/wt_baseline_gate.md", "scripts/c... |
