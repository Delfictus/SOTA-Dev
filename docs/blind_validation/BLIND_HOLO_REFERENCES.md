# BLIND HOLO REFERENCES
**Version:** 1.0  
**Written:** 2026-05-13 UTC — BEFORE any post-freeze holo coordinate access  
**Repo HEAD at writing:** HEAD (see git log)

---

## Blinding Status

This file was written **before** any predicted pocket coordinates (binding_sites.json) were compared against holo structures. The references listed below were selected based on published literature and public PDB metadata ONLY. No holo coordinates were loaded into visualization software or used to inspect binding sites.

Provisional entries (B03, B05) are flagged; their limitations are documented below.

---

## Holo Reference Table

| Slot | Target | Holo PDB 1 | Ligand 1 | Chain 1 | Holo PDB 2 | Ligand 2 | Chain 2 | Site | Notes |
|------|--------|-----------|---------|---------|-----------|---------|---------|------|-------|
| B01 | HRAS_Q61H | 6OIM | MOV (sotorasib/AMG510) | A | 7RPZ | 6IC (MRTX1133) | A | Switch II pocket | Both KRAS; SW2 pocket conserved ≥85% vs HRAS by sequence |
| B02 | CDK2_allosteric | 3PXZ | JWS (JWS648 allosteric) | A | 4GCJ | X64 (RC-3-89) | A | C-helix allosteric | JWS648 is established CDK2 allosteric inhibitor; X64 binding subsite TBD post-freeze |
| B03 | Kv1.2 | 2R9R | TM lipids | A | 2A79 | NAP | A | TM fenestration | PROVISIONAL — no confirmed drug-like small-molecule crystal structure for Kv1.2 fenestration pocket |
| B04 | MDM2 | 4ODF | 2U1 (compound 47) | A | 4ERF | 0R3 (compound 29) | A | p53-binding groove | Two independent MDM2 p53-mimetic inhibitor structures |
| B05 | TP53_apo | 3ZME | compound 23 | A | 4AGQ | compound 3 | A | Y220C cavity | PROVISIONAL — no published small-molecule crystal structure for WT p53 L1/H2 allosteric site; these are Y220C mutant references at a DIFFERENT pocket |
| B06 | cGAS | 4O67 | 1SY (cGAMP) | A | 5V8N | 8ZP (inhibitor) | A | Nucleotide-binding site | cGAMP = enzymatic product (MW 674 Da); 8ZP = synthetic inhibitor |
| B07 | TEAD1 | 3KYS | P1L (palmitate) | A | 5OAQ | MYR (myristate) | A | Palmitate/coactivator site | 3KYS = original pre-stripping holo; 5OAQ = TEAD4 same pocket type |
| B08 | CRBN | 4CI3 | Y70 (pomalidomide) | B | 5FQD | LVY (lenalidomide) | B | IMiD binding site | Both DDB1-CRBN complexes; chain B = CRBN in both |
| B09 | Thrombin_exosite | 1HAH | TYS (hirugen) | I | 3BEF | PAR1 fragment | B | Exosite I | Hirugen chain I occupies exosite I; PAR1 fragment contacts exosite I via hirudin-like sequence |
| B10 | ADRB2 | 3SN6 | P0G (BI167107) | R | 4LDE | P0G (BI167107) | A | Orthosteric (hard neg) | Active agonist-bound state; expected SR@5@8Å = 0 for cryptic pocket prediction |

---

## Per-target detail

### B01 — HRAS_Q61H (RAS-family switch II pocket)

**Primary: 6OIM** — KRAS G12C + AMG510 (sotorasib, covalent SW2 binder, MW~560 Da)  
- Sotorasib forms a covalent bond with C12 of KRAS G12C; occupies the switch II pocket (SII-P)  
- Chain A, ligand code MOV  

**Secondary: 7RPZ** — KRAS G12D + MRTX1133 (non-covalent SW2 binder, MW~604 Da)  
- MRTX1133 is a non-covalent, GDP-state SW2 pocket binder for KRAS G12D  
- Chain A, ligand code 6IC  

**Note:** Both references are KRAS (not HRAS). RAS family members share ≥85% sequence identity in the switch region; SW2 pocket topology is conserved. Scoring uses Kabsch alignment with seqid ≥0.50 threshold.

---

### B02 — CDK2_allosteric (C-helix / allosteric pocket)

**Primary: 3PXZ** — CDK2 + JWS648 + ANS (allosteric complex)  
- JWS648 (2-(4,6-DIAMINO-1,3,5-TRIAZIN-2-YL)-4-METHOXYPHENOL, MW~204 Da) binds CDK2 allosteric site  
- ANS (1-anilinonaphthalene-8-sulfonate, MW~299 Da) = fluorescent allosteric probe  
- Chain A, ligand codes JWS + 2AN  

**Secondary: 4GCJ** — CDK2 + RC-3-89 (X64, MW~437 Da)  
- X64 = 4-{[4-AMINO-5-(2-NITROBENZOYL)-1,3-THIAZOL-2-YL]AMINO}BENZENESULFONAMIDE  
- Chain A, ligand code X64  
- **Note:** binding subsite (allosteric vs ATP) to be confirmed post-freeze by structure inspection; scoring will report which PRISM4D pocket overlaps

---

### B03 — Kv1.2 (TM fenestration pockets) — PROVISIONAL

**Primary: 2R9R** — Kv1.2-Kv2.1 paddle chimera with beta subunit  
- TM region contains lipid/detergent molecules; no confirmed small-molecule drug in fenestration  
- Chain A  

**Secondary: 2A79** — Full-length Kv1.2 + Kvβ2 subunit  
- NAP (NADPH, MW~744 Da) is bound to the regulatory beta subunit T1 domain  
- Chain A (channel) / chain β  

**Limitation:** No published crystal structure with a drug-like small molecule in the Kv1.2 fenestration (side window) pocket exists as of 2026. If PRISM4D detects TM hydrophobic pockets, their coordinates will be reported; shell overlap against holo ligand positions will reflect the best available references. B03 may serve as a prospective prediction case pending future experimental validation.

---

### B04 — MDM2 (PPI p53-binding groove)

**Primary: 4ODF** — MDM2 (residues 17-111) + compound 47 (2U1, MW~467 Da)  
- Co-crystal at 2.0 Å; compound occupies Phe19/Trp23/Leu26 hotspot of p53-binding groove  
- Chain A, ligand code 2U1  

**Secondary: 4ERF** — MDM2 (residues 17-111) + compound 29 / AM-8553 precursor (0R3, MW~500 Da)  
- Chain A, ligand code 0R3  

---

### B05 — TP53_apo (WT p53 allosteric, L1/H2 loop) — PROVISIONAL

**Target site:** L1/H2 allosteric pocket in WT p53 core domain  

**Primary: 3ZME** — p53 Y220C mutant + compound 23 (PhiKan series)  
- Chain A. Compound binds the Y220C cavity, which is structurally distinct from the L1/H2 allosteric site.  
- **Not the target site** — used only as a structural reference for scoring vicinity  

**Secondary: 4AGQ** — p53 Y220C mutant + compound 3 (PhiKan series)  
- Chain A. Same caveat as 3ZME.  

**Limitation:** As of 2026, no published crystal structure exists with a small-molecule ligand at the WT p53 L1/H2 allosteric pocket. The Y220C-specific compounds in 3ZME/4AGQ bind a different pocket on the opposite face of the β-sandwich. B05 is expected to test PRISM4D's ability to predict a novel cryptic site pending future crystallographic confirmation. If PRISM4D predicts a pocket at L1/H2, this will be reported as a prospective novel prediction.

---

### B06 — cGAS (nucleotide-binding site)

**Primary: 4O67** — Human cGAS + cGAMP (enzymatic product, 1SY, MW~674 Da)  
- Crystal at 2.3 Å; cGAMP = cyclic-di-GMP-AMP, the second messenger product of cGAS  
- Chain A, ligand code 1SY  

**Secondary: 5V8N** — Human cGAS + high-affinity inhibitor 8ZP  
- Discovery paper for synthetic cGAS inhibitor  
- Chain A, ligand code 8ZP  

---

### B07 — TEAD1 (palmitate/coactivator site)

**Primary: 3KYS** — TEAD1 + P1L (S-palmitoyl-cysteine, covalently attached at Cys344)  
- This is the ORIGINAL PDB before P1L was stripped for the apo structure  
- P1L coordinates define the target palmitate-binding site  
- Chain A, HETATM P1L at residue 344  
- **Known leakage:** This structure was used to generate the apo (P1L was stripped), so the operator knew P1L existed at Cys344. Documented in OPEN_QUESTIONS_AND_MISSING_ARTIFACTS.md as B07 unavoidable leakage.  

**Secondary: 5OAQ** — TEAD4 + YAP peptide + myristate (MYR, covalently bound)  
- Myristate occupies the equivalent fatty acid binding pocket in TEAD4  
- Chain A, ligand code MYR  

---

### B08 — CRBN (IMiD binding site)

**Primary: 4CI3** — DDB1-CRBN + pomalidomide (Y70, MW~273 Da)  
- Crystal at 3.5 Å; pomalidomide binds the tri-Trp cavity of CRBN  
- Chain B (= CRBN; chain A = DDB1), ligand code Y70 at residue 1429  

**Secondary: 5FQD** — DDB1-CRBN-CK1α + lenalidomide (S-lenalidomide LVY, MW~259 Da)  
- Crystal at 3.1 Å; lenalidomide + CK1α substrate  
- Chain B (= CRBN), ligand code LVY at residue 1438  

---

### B09 — Thrombin_exosite (Exosite I)

**Primary: 1HAH** — Thrombin + PPACK (active site) + hirugen (exosite I)  
- Hirugen = C-terminal 13-residue fragment of hirudin, occupies fibrinogen-recognition exosite I  
- Modeled as chain I; TYS (sulfotyrosine, MW~261 Da for Tyr+SO3) at exosite I  
- Chain H = heavy chain (our prediction target)  

**Secondary: 3BEF** — Thrombin + PAR1 extracellular fragment  
- PAR1 N-terminal hirudin-like sequence contacts exosite I  
- Chain B = PAR1 fragment  

**Note:** Thrombin exosite I is historically targeted by peptide inhibitors (hirudin, bivalirudin). No validated small-molecule-only crystal structure for exosite I has been identified. Hirugen (TYS-containing peptide) and PAR1 fragment both define the exosite I site crystallographically with atoms >150 Da at the target site.

---

### B10 — ADRB2 (GPCR hard negative)

**Primary: 3SN6** — β2AR + Gs protein complex + BI167107 (P0G, MW~461 Da)  
- Active agonist-bound state; BI167107 is at the orthosteric site  
- Chain R (= β2AR in the Gs complex), ligand code P0G  

**Secondary: 4LDE** — β2AR + BI167107 + nanobody  
- Second independent active-state agonist-bound structure  
- Chain A (= β2AR), ligand code P0G  

**Hard negative criterion:** Expected SR@5@8Å = 0 for orthosteric holo references when scored against cryptic pocket predictions. Shell overlap with orthosteric site BI167107 IS expected (the orthosteric site is preformed and visible in apo). Cryptic pocket detection should be absent.

---

## Scoring protocol

All holos are scored by `prism_pub_baseline_validator.py` post-freeze using:
- Kabsch alignment: apo chain → holo chain (seqid ≥ 0.50, RMSD ≤ 5.0 Å)
- Ligand heavy atoms: all HETATM records with the holo ligand code, filtered to ≥6 atoms
- Shell radii: 4 Å, 6 Å, 8 Å
- SR@k: success if ≥1 of top-k predicted sites has centroid within shell distance of any holo ligand heavy atom

For targets with peptide-based holo references (B09, B07), all heavy atoms of the ligand chain/modified residue are used.

---

## Chain of custody

This file was created before any post-freeze holo coordinate access. Its SHA256 hash is recorded in the GLOBAL_PREDICTION_FREEZE commit. Any modification after freeze would be detectable from the git history.
