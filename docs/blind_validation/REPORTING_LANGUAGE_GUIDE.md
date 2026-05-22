# REPORTING LANGUAGE GUIDE
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Core platform framing (use verbatim)

These 18 elements must be preserved in all reporting. Do not substitute synonyms that alter meaning.

| Element | Correct phrasing | Incorrect variants |
|---------|-----------------|-------------------|
| Simulation type | "perturbation-driven molecular dynamics" | "enhanced sampling MD", "steered MD" |
| State | "apo-state candidate generation" | "unbound structure", "ligand-free simulation" |
| Detection method | "fused spike sensing" | "post-run pocket detection", "trajectory analysis" |
| Protocol name | "CryoUV fast_35k hysteresis protocol" | "temperature cycling", "annealing" |
| Step count | "45,000 steps per stream" | "55,000 steps" (WRONG — see METHOD_DRIFT C1) |
| Streams | "8 parallel streams" (standard) or "20 streams" (>400 res) | "replicates", "trajectories" |
| Engine | "PRISM4D engine" | "PRISM", "the algorithm" |
| Sensor | "device-resident sensing kernel" | "GPU-side detector", "CUDA kernel" |
| Output | "spike events" | "peaks", "pulses", "signals" |
| Sites | "candidate sites" or "binding sites" | "predicted pockets", "putative pockets" |
| Ranking | "lexicographic ranking (persistence → pass_fraction → stability → quality)" | "scoring function", "composite score" |
| Therm | "thermodynamic classification (CRYPTIC / ORTHOSTERIC / ALLOSTERIC)" | "cryptic site category" |
| Gating | "GTCKL+RS gating stack" | "quality filter", "confidence filter" |
| Validation | "prospective-retrospective blind computational validation" | "retrospective validation", "benchmark" |
| Freeze | "prediction freeze before holo coordinate access" | "blinded analysis" |
| Baselines | "fpocket v4.2.3 and P2Rank v2.4.2 under default parameters" | "standard pocket finders" |
| Scoring | "ligand-shell scoring at 4/6/8 Å from holo ligand heavy atoms after Kabsch alignment" | "overlap with bound structure" |
| Null | "pair-breaking null permutation" | "random baseline", "permutation test" |

---

## Claims: what can and cannot be asserted

### Can assert (with evidence)
- "PRISM4D detected a candidate site whose residues overlap with the holo ligand shell at [X] Å in [N]/[total] validation structures."
- "PRISM4D ranked this site within the top-[k] predictions."
- "The PRISM4D site co-localizes with the binding site of [drug class] in [holo PDB]."
- "PRISM4D outperforms fpocket and P2Rank at SR@5@8Å ([X]% vs [Y]% vs [Z]%)."

### Cannot assert (without further evidence)
- "PRISM4D predicts this site is druggable." (Can say: therm_class=CRYPTIC, consistent with druggability.)
- "This is a cryptic pocket." (Can say: the site was not detectable in the apo PDB by geometry-only methods and opened during CryoUV perturbation.)
- "PRISM4D is better than AlphaFold." (AlphaFold is not a pocket detection method; do not compare.)
- "This site can be targeted by [specific compound class]." (Can say: the PRISM design brief suggests [anchor/growth vector].)
- Statistical significance without null control p-values.

---

## Method deviation language

When deviations from pub run exist, state:
"The blind validation adds --replica-seed 42 for internal reproducibility, not present in the original publication runs. All other flags match the locked publication methodology (see LAST10_METHOD_LOCK.md §2)."

---

## Uncertainty hedges

Use exactly when uncertain:
- "estimated" for unmeasured quantities
- "consistent with" for indirect evidence
- "further investigation is required to confirm" for hypotheses
- Do NOT use: "suggests", "indicates", "implies" without qualifying
