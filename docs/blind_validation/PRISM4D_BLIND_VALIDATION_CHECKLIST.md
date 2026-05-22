# BLIND VALIDATION EXECUTION CHECKLIST
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Phase 1: Structure preparation

- [ ] B01 HRAS_Q61H — 4L9S downloaded
- [ ] B01 HRAS_Q61H — prism-clean.py passed (residue diversity ≥15)
- [ ] B01 HRAS_Q61H — topology.json generated
- [ ] B01 HRAS_Q61H — nma_modes.json generated
- [ ] B02 CDK2_allosteric — 1HCL downloaded + cleaned + topology + NMA
- [ ] B03 Kv1.2 — 3LUT downloaded + cleaned + topology + NMA
- [ ] B04 MDM2 — 1Z1M downloaded + cleaned + topology + NMA
- [ ] B05 TP53_R175H — 2OCJ downloaded + cleaned + topology + NMA
- [ ] B06 cGAS — 4KM5 downloaded + cleaned + topology + NMA
- [ ] B07 TEAD1 — 3KYS downloaded + cleaned + topology + NMA
- [ ] B08 CRBN — 4TZ4 downloaded + cleaned + topology + NMA
- [ ] B09 Thrombin_exosite — 1HAH downloaded + cleaned + topology + NMA
- [ ] B10 ADRB2 — 2RH1 downloaded + cleaned + topology + NMA
- [ ] Leakage scan: no drug-like HETATM in any clean PDB

## Phase 2: Engine runs

- [ ] B01 HRAS_Q61H — run complete, binding_sites.json non-empty
- [ ] B02 CDK2_allosteric — run complete
- [ ] B03 Kv1.2 — run complete (--multi-stream 20)
- [ ] B04 MDM2 — run complete
- [ ] B05 TP53_R175H — run complete
- [ ] B06 cGAS — run complete
- [ ] B07 TEAD1 — run complete
- [ ] B08 CRBN — run complete
- [ ] B09 Thrombin_exosite — run complete
- [ ] B10 ADRB2 — run complete (--multi-stream 20)

## Phase 3: Baselines

- [ ] fpocket: all 10 targets complete
- [ ] P2Rank: all 10 targets complete

## Phase 4: Freeze

- [ ] Pre-freeze leakage scan passed (all clean)
- [ ] B01–B10 artifacts copied to frozen/
- [ ] B01–B10 SHA256 manifests generated
- [ ] B01–B10 frozen dirs chmod 555
- [ ] B01–B10 PREDICTION_FREEZE_*.md written
- [ ] GLOBAL_PREDICTION_FREEZE.md assembled
- [ ] Git commit made (freeze timestamp in history)

## Phase 5: Post-freeze scoring

- [ ] SHA256 manifests verified for all 10 (before opening holos)
- [ ] prism_pub_baseline_validator.py run for all 10 targets
- [ ] prism_vs_holo.csv generated for all 10
- [ ] baseline_vs_holo.csv generated for all 10
- [ ] AGGREGATE_PRISM_VS_HOLO.csv assembled
- [ ] AGGREGATE_BASELINE_VS_HOLO.csv assembled

## Phase 6: LORO + Family

- [ ] loro_results.csv generated for all targets with ≥2 holo refs
- [ ] family_results.csv generated for all targets
- [ ] RR@LORO computed

## Phase 7: Null controls

- [ ] pair_breaking_null.py run (or gap documented)
- [ ] empirical p-value computed (or gap documented)

## Phase 8: Visualization

- [ ] pymol_overlay.pml generated for primary targets (B01–B09)
- [ ] Representative figures captured

## Phase 9: Final report

- [ ] SR@k table populated (1/3/5 × 4/6/8Å)
- [ ] LORO table populated
- [ ] Family collapse table populated
- [ ] Baseline comparison included
- [ ] Hard negative (B10 ADRB2) result documented
- [ ] Null control results or gap noted
- [ ] BLIND_VALIDATION_FINAL_REPORT.md written
- [ ] Report committed to docs/blind_validation/reports/
