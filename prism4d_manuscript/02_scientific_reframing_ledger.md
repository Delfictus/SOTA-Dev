# PRISM-4D Scientific Reframing & Vernacular Ledger

**Purpose**: Map every marketing/promotional phrase in v3.3 to proper scientific language.
**Rule**: Every substitution must preserve technical accuracy while meeting peer-review tone standards.

---

## CRITICAL REMOVALS (no scientific equivalent -- delete entirely)

| Location | Current Text | Action | Rationale |
|----------|-------------|--------|-----------|
| Title page | "SNDC v9 Engine" | DELETE | Product versioning; use "PRISM-4D" as method name |
| Title page | "SAM.gov Registered" | DELETE | Government contracting credential, not science |
| Title page | "Active UEI & CAGE" | DELETE | Same as above |
| Title page | "Commit b7a36179" | DELETE (until open-source) | Unverifiable private repo commit |
| p.2 subtitle | "A neuromorphic molecular dynamics platform for drug-targetable pocket discovery" | DELETE | Marketing subtitle; journals don't use this format |
| p.9 | "Hero Target 1:" | DELETE prefix | Industry jargon; use "Case Study 1:" |
| p.9 | "Hero Target 2:" | DELETE prefix | Same; use "Case Study 2:" |
| p.15 | "$305M therapeutic opportunity" | DELETE | Financial projection, not science |
| p.15 Table 8 | Green-colored PRISM-4D text | REFORMAT | Uniform formatting for all methods |
| p.17 | "the future of computational drug target discovery is neuromorphic, physics-first, and democratized" | DELETE | Company tagline, not conclusion |

---

## TONE REFRAMING (marketing -> scientific)

| # | Location | Current (Marketing) | Reframed (Scientific) | Category |
|---|----------|--------------------|-----------------------|----------|
| 1 | Abstract | "100% detection accuracy at <10A DCC across all 11 ground-truth targets" | "All 11 ground-truth targets were detected within 10A DCC (range: 3.6-9.8A); 4/11 (36%) achieved <5A" | INFLATED_CLAIM |
| 2 | Abstract | "democratizes cryptic pocket detection for the global research community" | "enables cryptic pocket detection on consumer-grade hardware without external software dependencies" | ADVOCACY |
| 3 | Abstract | "making capabilities previously restricted to institutions with supercomputer access or six-figure software budgets available to any laboratory with a desktop GPU" | DELETE (redundant with reframed claim above) | ADVOCACY |
| 4 | Abstract | "~$999" | DELETE from abstract; mention once in Methods hardware section | PRICING |
| 5 | p.3 | "structural inequality in drug discovery" | "unequal access to computational infrastructure" | ADVOCACY |
| 6 | p.3 | "eleven techniques without precedent in the pocket detection literature" | "several techniques not previously applied to pocket detection" | INFLATED_CLAIM |
| 7 | p.5 | "No prior pocket detection method employs wavelength-specific UV perturbation for cryptic site discovery" | "To our knowledge, wavelength-specific UV perturbation has not been previously applied to computational pocket detection" | OVERCLAIM |
| 8 | p.6 | "This 5-phase non-equilibrium protocol represents a paradigm shift from conventional MD-based pocket detection" | "This non-equilibrium protocol differs from conventional equilibrium MD approaches to pocket detection" | HYPERBOLE |
| 9 | p.7 | "This is, to our knowledge, the first application of ray tracing hardware acceleration to any molecular analysis problem" | "We apply ray tracing hardware acceleration to molecular spatial queries, which to our knowledge has not been previously reported" | OVERCLAIM |
| 10 | p.7 | "No other pocket detection tool combines cosolvent-guided hotspot identification with active UV energy deposition" | "This approach combines cosolvent-guided identification with active UV energy deposition, which has not been previously described" | OVERCLAIM |
| 11 | p.9 | "Gold-standard cryptic benchmark" (Table 1, 1BTL) | "Established cryptic pocket benchmark" | HYPERBOLE |
| 12 | p.13 | "The core innovation of PRISM-4D is reframing pocket detection as a thermodynamic event detection problem" | "PRISM-4D approaches pocket detection as a thermodynamic event detection problem" | OVERCLAIM |
| 13 | p.14 | "constitute a new computational paradigm for structure-based drug design" | "represent a distinct computational approach to structure-based pocket detection" | HYPERBOLE |
| 14 | p.15 | "potentially transformative finding" | "computationally predicted cryptic pocket requiring experimental validation" | HYPERBOLE |
| 15 | p.15 | "The entire PRISM-4D platform ... runs on a single consumer-grade NVIDIA RTX 5080 GPU" | Keep, but state once in Methods, not repeated in Discussion | REPETITION |
| 16 | p.15 | "democratization of cryptic pocket detection capability" (bolded) | "accessibility of cryptic pocket detection" (unbolded) | ADVOCACY |
| 17 | p.15 | "$0 per run" | "no per-run compute costs" | PRICING |
| 18 | p.16 | "30x cheaper" (Figure 11 annotation) | Remove annotation; let data speak | MARKETING |
| 19 | p.16 | "supercomputer-grade pocket detection" | "physics-simulation-based pocket detection" | HYPERBOLE |
| 20 | p.17 | "sub-angstrom reproducibility" | "centroid displacement <0.06A across 5 stochastic seeds (beta-catenin Site 6)" | MISLEADING_PRECISION |

---

## TERMINOLOGY CORRECTIONS

| Current Term | Issue | Corrected Term | Rationale |
|--------------|-------|----------------|-----------|
| "Neuromorphic" (title, throughout) | Implies brain-inspired hardware | "Spike-event-driven" or "LIF-based event detection" | Standard neuromorphic computing refers to Loihi/SpiNNaker architectures |
| "Hero Target" | Industry demo jargon | "Case Study" or "Validation Target" | Standard scientific terminology |
| "Innovation Stack" (Sec 5.2, Fig 10) | Product marketing term | "Technical Contributions" or "Methodological Components" | Scientific framing |
| "Democratization" | Political/advocacy term | "Accessibility" or "Reduced hardware requirements" | Neutral scientific language |
| "Paradigm shift" | Kuhnian philosophy, overused | "Distinct approach" or "Alternative framework" | Reviewers react negatively to this phrase |
| "Air-gapped operation" | InfoSec jargon | "Operates without network connectivity" | More accessible to bio audience |
| "Pharma-grade" (if used) | Industry marketing | "Publication-quality" or omit | Not a defined scientific standard |

---

## UNDEFINED TERMS REQUIRING DEFINITION

| Term | First Appearance | Definition Needed |
|------|-----------------|-------------------|
| DCC (Distance from Centroid to Centroid) | Abstract | Define at first use: "DCC, the Euclidean distance between the predicted site centroid and the crystallographic ligand centroid after Kabsch superposition" |
| SNDC | Abstract | "Spike-Native Density Clustering (SNDC)" at first use |
| LIF | Abstract | "leaky integrate-and-fire (LIF)" at first use in abstract |
| BVH | p.4 | "bounding volume hierarchy (BVH)" at first use |
| EFP | p.4 | Already defined on p.4; move definition earlier if mentioned in abstract |
| CCNS | p.6 | "Conformational Crackling Noise Spectroscopy (CCNS)" -- currently appears without any expansion |
| Quality score | Throughout results | Must define formula in Methods |
| Druggability score | Throughout results | Must define formula in Methods |
| ARM repeats | Table 4 | "armadillo (ARM) repeats" |

---

## CLAIMS REQUIRING QUALIFICATION

| Claim | Current Strength | Required Qualification |
|-------|-----------------|----------------------|
| "100% detection accuracy" | Unqualified | "at a 10A DCC threshold; 36.4% at the more stringent 5A threshold used by CryptoSite" |
| "Novel cryptic pocket" (beta-catenin) | Presented as discovery | "Computationally predicted novel pocket; experimental validation required" |
| "85-90% accuracy" (water inference) | Unqualified | Must cite validation dataset or remove claim |
| "10-60x speedup" (RT cores) | Unqualified | Must provide timing benchmark or qualify as "estimated" |
| "No proprietary software dependencies" | Unqualified | "Requires only the NVIDIA CUDA runtime (proprietary but freely available)" |
| "11 techniques without precedent" | Unqualified | Reclassify: ~5 genuinely novel, ~3 novel applications of known techniques, ~3 engineering optimizations |
| "First application of RT hardware" | Absolute claim | "To our knowledge, the first..." with literature search caveat |

---

## WORD FREQUENCY FLAGS

| Word/Phrase | Count in v3.3 | Max Acceptable | Action |
|-------------|---------------|----------------|--------|
| "democratize/democratization" | ~5 | 1 | Use once in Discussion; replace others with "accessibility" |
| "without precedent" | ~4 | 1 | Use once with qualification; replace others |
| "consumer GPU" / "$999" | ~6 | 2 | Mention in Methods (hardware) and once in Discussion |
| "novel" | ~8 | 3 | Reserve for genuinely novel contributions |
| "innovation" | ~5 | 2 | Replace most with "technique" or "approach" |
| "first" (as in "first to...") | ~5 | 2 (qualified) | Always prefix with "to our knowledge" |
