# PGx RAW DATA — DSTW INTEGRATION DIRECTIVES & SUPPLEMENTAL RECOMMENDATIONS
# This document describes HOW the raw data files feed into the existing
# PRISM-4D pipeline. It does NOT implement classification logic — that
# belongs to the Thermodynamic Observatory and Rust Oracle.

## FILES IN THIS PACKAGE

1. gnomAD_GLP1R_missense_variants.csv
   - 22 real human GLP1R missense variants
   - Real MAFs from gnomAD v4.1 (807K individuals)
   - Real functional annotations from published assays
   - Real population-stratified allele frequencies (EUR/AFR/EAS/SAS/AMR)
   - Column: pdb_chain_residue_6xox — maps each variant to the 6XOX
     structure coordinate system the Rust Oracle already uses

2. GLP1R_structural_domain_map.csv
   - 17 structural domains with residue boundaries
   - Ballesteros-Weinstein numbering where applicable
   - PDB 6XOX resolution status per domain
   - Structural role annotations

3. GLP1R_cross_species_conservation.csv
   - Human/NHP/Rat/Mouse/Dog residue alignment at key positions
   - Conservation scores (0-1)
   - Pocket contact and allosteric relevance flags
   - Feeds directly into the Phase 2D Variant Grid species selectivity
     analysis already defined in the SOT

4. GLP1R_functional_assay_clusters.csv
   - Published cluster assignments from Gao et al. Nat Metab 2023
   - Quantitative pathway data (cAMP Emax %, potency fold-shift,
     β-arr2 Emax %, ERK Emax %, Ca2+ Emax %)
   - Surface/total expression percentages
   - Phenotype scores

5. SOURCE_BIBLIOGRAPHY.md
   - Full provenance chain for every data point
   - DOIs, journal citations, exact findings

## DSTW PIPELINE INTEGRATION POINTS

### STAGE 1: Ingestion into Variant Grid
- Target: scripts/patch_phase_2d_manifest.py (already exists in FILE_LEDGER)
- gnomAD_GLP1R_missense_variants.csv should be ingested alongside the
  existing Rat_GLP1R_Homology and Dog_GLP1R_Homology conditions
- The cross_species_conservation.csv provides the alignment data for
  the non-conserved pocket residues (R190, L352, etc.) that explain
  species selectivity
- Each variant becomes a new condition in the Phase 2D manifest:
  Human_GLP1R_{mutation} (e.g., Human_GLP1R_A316T)

### STAGE 2: TDMS via Thermodynamic Observatory
- For each ingested variant, the system should autonomously:
  1. Generate mutant 3D topology (in silico mutagenesis on 6XOX)
  2. Run the 5-phase CCNS hysteresis protocol (80 replicas)
  3. Compute signal_grid_variance_channel for the mutant
  4. Compute translation_pathway_nodes for the mutant
  5. Compare mutant tensors vs WT tensors
- This is what the existing PRISM-4D engine already does — the variant
  CSV is just new input conditions, not new logic

### STAGE 3: Intersection (Rust Oracle — existing logic)
- The Rust vspace_pruner already maps atoms → voxel_idx → signal_grid
- For PGx analysis, the same mapping applies but to MUTANT signal grids
- The Three-Tier Guillotine already classifies voxels as:
  stable_occupied, thermally_destabilized, thermally_activated, void
- Comparing WT guillotine output vs MUTANT guillotine output reveals
  exactly which mutations break the allosteric cascade
- The functional_assay_clusters.csv provides external validation:
  if the engine classifies a variant as allosteric-severing, and
  Gao et al. measured complete cAMP LoS for that same variant,
  that's concordance. If they disagree, that's a calibration signal.

### STAGE 4: Output — PGx Manifest
- The system's own classification outputs (not hardcoded labels)
  should be formatted into the Phase 3 exclusion manifest
- Failure modes should emerge from the physics (which thermal phases
  are disrupted, which voxel classifications shift) not from
  pre-assigned categories

## SUPPLEMENTAL DATA RECOMMENDATIONS (FOR CONSIDERATION)

### A. AlphaMissense Pathogenicity Scores
- Google DeepMind's AlphaMissense (Science 2023) provides per-residue
  pathogenicity predictions for every possible missense mutation
- Available via: https://alphamissense.hegelab.org/ or direct download
- Would provide predicted pathogenicity for ALL 463×19 = 8,797 possible
  GLP1R missense mutations, not just the ~22 with published data
- Integration: additional column in the variant CSV, or separate lookup
  table that the engine queries during TDMS

### B. ClinVar Clinical Significance Annotations
- ClinVar contains clinical significance classifications for some
  GLP1R variants (Pathogenic/Likely Pathogenic/VUS/Benign)
- Would add a regulatory-grade clinical annotation layer
- Integration: join on rsID

### C. gnomAD Constraint Metrics for GLP1R
- Gene-level constraint (pLI, LOEUF, missense Z-score) from gnomAD v4
- GLP1R's constraint scores would tell the engine how tolerant the gene
  is to missense variation overall — context for interpreting rare variants
- Available in gnomAD gene constraint table

### D. GTEx Expression Data
- Tissue-specific GLP1R expression levels (pancreas, brain, gut, heart)
- Would allow the engine to weight variant impact by tissue relevance
- A variant that disrupts signaling in pancreatic β-cells matters more
  for GLP-1RA efficacy than one affecting brain expression

### E. Extended gnomAD Variant Pull (Full Gene)
- The 22 variants here are the ones with published functional data or
  notable MAF. gnomAD v4.1 contains hundreds more GLP1R missense
  variants at ultra-rare frequencies (MAF < 0.001%)
- A full VCF slice of GLP1R from gnomAD would give the engine every
  known human variant, not just the characterized ones
- Download: gnomAD VCF for chr6:39,016,557-39,059,079 (GRCh38)
- This is the highest-value supplemental dataset — it would let the
  TDMS protocol run on variants that have ZERO published functional
  data, generating de novo thermodynamic predictions that could be
  validated against future experimental work

### F. UK Biobank Phenotype Associations
- Gao et al. 2023 showed associations between GLP1R surface expression
  variants and HbA1c/BMI in 200K UK Biobank participants
- The individual-level association data (effect sizes, p-values per
  variant per phenotype) would be valuable calibration targets
- The engine's TDMS predictions could be benchmarked against these
  real-world phenotype associations

### G. Biased Agonist Response Data
- Hinds et al. 2024 showed that D344E responds differently to biased
  (SRB107) vs unbiased (exendin-4, semaglutide) agonists
- Since aleniglipron's mechanism involves biased agonism (the PRISM
  FRAG-A steric wedge blocks β-arrestin coupling), variant-specific
  response to biased agonists is directly relevant
- This data would feed into the bifurcated reward function's
  w_bias·Π_clash(lock) term — some human variants may enhance or
  abolish the biased signaling that makes aleniglipron work
