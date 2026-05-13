# OPEN QUESTIONS AND MISSING ARTIFACTS
**Version:** 1.0  
**Generated:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5

---

## Missing artifacts (blocking or non-blocking)

| Artifact | Status | Blocking? | Resolution |
|----------|--------|-----------|-----------|
| Pair-breaking null script | MISSING | NO (report gap) | Implement scripts/quarantine/null_controls/pair_breaking_null.py per DECOY_AND_NULL_CONTROL_PROTOCOL.md |
| Permutation null script | MISSING | NO | Same |
| Family recurrence null script | MISSING | NO | Same |
| Standalone LORO script | MISSING (embedded in validator) | NO | Use prism_pub_baseline_validator.py |
| M4R pub run output | NOT FOUND at expected path | NO | Author to verify alternate path |
| Blind validation figure generation script | MISSING | NO | Write scripts/quarantine/generate_blind_validation_figures.py |
| Blind validation aggregate report script | MISSING | NO | Write scripts/quarantine/generate_blind_validation_report.py |

---

## Open questions requiring author input

| Question | Impact | Who |
|----------|--------|-----|
| Fix PRISM4D_Complete_Technical_Reference.md lines 237-238, 249: 55,000→45,000 steps | HIGH — doc cited in preprint may confuse reviewers | Author |
| Confirm 7ATA p53 holo reference PDB ID (HTTP 404 in prior run) | MEDIUM — only affects publication p53_Y220C post-hoc scoring | Author |
| Confirm M4R status: is it in the preprint or was the run discarded? | LOW | Author |
| Were all 10 pub run targets run with --nma-perturb? (only confirmed for 9) | MEDIUM — affects method parity with blind runs | Author |
| Does the v1.1-frozen build support --multi-differential? | LOW — omitted from blind runs regardless | Author |

---

## Open questions: blind validation targets

| Question | Impact | Resolution |
|----------|--------|-----------|
| 2OCJ (TP53_R175H): verify this is a true apo of the R175H mutant | HIGH — wrong mutant would invalidate B05 | Verify SEQRES record for His175 mutation |
| 2RH1 (ADRB2): T4L fusion residues — confirm correct chain selection | MEDIUM — T4L creates false pockets | prism-clean.py keeps chain A; verify topology offset |
| 4L9S (HRAS): confirm Q61H mutation present in SEQRES | HIGH | Verify before prep |
| 1HAH (thrombin): confirm exosite I is unoccupied | MEDIUM | Check HETATM records at exosite I |

---

## Known documentation gaps (author corrections needed)

| File | Location | Issue | Correction |
|------|----------|-------|-----------|
| PRISM4D_Complete_Technical_Reference.md | L237-238 | "14,000" cold_return, "55,000 total" | 4,000 cold_return, 45,000 total |
| PRISM4D_Complete_Technical_Reference.md | L249 | "engine.run(55000 steps)" | 45,000 |
| CLAUDE.md canonical command | global CLAUDE.md | --spike-percentile 70 shown as canonical | 50 is dominant pub run practice |
| CLAUDE.md canonical command | global CLAUDE.md | Missing --nma-perturb from canonical | Should include --nma-perturb |

---

## Performance unknowns

| Unknown | Impact |
|---------|--------|
| Whether CDK2 allosteric site opens in 55k-step protocol | B02 ran 55k steps; detected 3 Cryptic sites at ranks 1-3 (RESOLVED: sites detected) |
| Whether Kv1.2 fenestration pockets require >20 streams | B03 RAN WITH 8 STREAMS (not 20 per protocol) — see Deviations below |
| Whether 4TZ4 (CRBN apo) has thalidomide site clearly apo | Verified: 4TZ4 chain C has LVY (lenalidomide) stripped in prep; apo confirmed |
| ADRB2 T4L false positives | Expected; document and exclude from hard-neg scoring |

---

## Execution deviations (2026-05-13)

| Deviation | Impact | Resolution |
|-----------|--------|-----------|
| B03 Kv1.2 run with 8 streams (protocol: 20) | Reduced spatial sampling of TM region | Accept 8-stream result; note deviation; may re-run if no TM pockets detected |
| B05 labeled "TP53_R175H" in output dirs/logs | Nominal only — structure is 2OCJ (WT p53 R282Q context but used as WT apo) | Report corrects to "TP53_apo (2OCJ, WT p53 core domain)" |
| B07 TEAD1 unavoidable holo leakage | P1L (palmitate) was stripped from 3KYS to make apo → operator knew palmitate site | Documented; holo reference = 3KYS (original). Score will reflect self-referential nature. |
| B05 no validated small-molecule holo for L1/H2 site | WT p53 L1/H2 allosteric site has no published crystal structure with small molecule | B05 treated as novel cryptic prediction; Y220C structures (3ZME/4AGQ) used as provisional holos at a different pocket |
| B09 Thrombin exosite I holos are peptide-based (not small molecule) | Hirugen (1HAH chain I) and PAR1 fragment (3BEF) are peptides, not drug-like SM | Acceptable: peptide ligand defines site geometry; hirugen TYS MW>150 Da; documented limitation |
