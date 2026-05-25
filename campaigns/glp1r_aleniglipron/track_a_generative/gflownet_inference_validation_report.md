# GFlowNet v1 — Inference Audit Validation Report

Generated: 2026-05-24T13:58:33Z
Summary: OK=31  WARN=0  FAIL=0
Verdict: **PASS**

## Findings

- [OK] required file present: model (1,036,389 B)
- [OK] required file present: manifest (5,487 B)
- [OK] required file present: raw_samples (198,553 B)
- [OK] required file present: consensus_scores (36,824 B)
- [OK] required file present: filtered (48,897 B)
- [OK] required file present: top100_parquet (13,857 B)
- [OK] required file present: top100_csv (27,865 B)
- [OK] required file present: top100_md (13,529 B)
- [OK] required file present: top25_high (10,375 B)
- [OK] required file present: top25_expl (10,502 B)
- [OK] required file present: medchem_report (599 B)
- [OK] required file present: baseline_json (1,107 B)
- [OK] required file present: baseline_md (727 B)
- [OK] required file present: audit_json (2,120 B)
- [OK] required file present: audit_md (1,702 B)
- [OK] required file present: failure_matrix (932 B)
- [OK] required file present: review_cards_html (36,314 B)
- [OK] required file present: review_cards_md (32,453 B)
- [OK] top100 unique SMILES = 100 (rows=100)
- [OK] top100 heavy_atom_count floor = 12 (require >=8)
- [OK] top100 max anchor share = 1.0% (require <25%)
- [OK] top100 OBSERVED labels = 0 (must be 0)
- [OK] consensus rows with oracle_valid_all=false = 0
- [OK] plot present: reward_distribution.png
- [OK] plot present: reward_vs_uncertainty.png
- [OK] plot present: reward_vs_clash.png
- [OK] plot present: cryptic_bonus_vs_reward.png
- [OK] plot present: top100_cluster_summary.png
- [OK] plot present: temperature_source_distribution.png
- [OK] plot present: trajectory_entropy_distribution.png
- [OK] no forbidden overclaim phrases in any narrative MD
