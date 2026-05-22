# GLP-1R WT Baseline Gate

Prepared by `scripts/prism-prep` using mmCIF `_struct_ref*` UniProt
alignment records plus optional PDBFixer/OpenMM apo-priming.
Construct/fusion segments, partners, ligands, and target-sequence
differences are gated before any PRISM-DSTW calibration use.

| PDB | role | WT baseline eligible | apo-prime status | point reversions | unresolved loop/linker | minimization | clearance |
|---|---|---:|---|---:|---:|---|---|
| `5VEX` | inactive/NAM-bound quiet-thermal control | `False` | `clean` | 10/10 | 0 | not_applicable_kinematic | cleared: apo-primed, WT-reverted, strict geometry gate passed |
| `6X1A` | active/non-peptide agonist + Gs productive activation control | `False` | `clean` | 1/1 | 0 | pass | cleared: apo-primed, WT-reverted, strict geometry gate passed |

## Apo-Priming Outputs

| PDB | stripped target-only scaffold | point-reverted scaffold | campaign input path |
|---|---|---|---|
| `5VEX` | `/home/diddy/Desktop/PRISM-DSTW/prism-dstw-calibration/campaigns/glp1r_aleniglipron/topology/5VEX/apo_primed/5VEX_target_only_apo_stripped_monomer_A.pdb` | `/home/diddy/Desktop/PRISM-DSTW/prism-dstw-calibration/campaigns/glp1r_aleniglipron/topology/5VEX/apo_primed/5VEX_target_only_point_reverted.pdb` | `/home/diddy/Desktop/PRISM-DSTW/prism-dstw-calibration/campaigns/glp1r_aleniglipron/inputs/structures/5VEX_apo_primed_wt_attempt.pdb` |
| `6X1A` | `/home/diddy/Desktop/PRISM-DSTW/prism-dstw-calibration/campaigns/glp1r_aleniglipron/topology/6X1A/apo_primed/6X1A_target_only_apo_stripped.pdb` | `/home/diddy/Desktop/PRISM-DSTW/prism-dstw-calibration/campaigns/glp1r_aleniglipron/topology/6X1A/apo_primed/6X1A_target_only_point_reverted.pdb` | `/home/diddy/Desktop/PRISM-DSTW/prism-dstw-calibration/campaigns/glp1r_aleniglipron/inputs/structures/6X1A_apo_primed_wt_attempt.pdb` |

## Required Operator Interpretation

- `clean` means target-only, ligand/partner-stripped, point-mutated
  back to canonical GLP-1R where supported, and either
  OpenMM-minimized or strict kinematic-CCD geometry validated.
- `blocked_pending_loop_rebuild` means a usable stripped scaffold was
  written, but at least one deletion/linker/nonstandard residue still
  requires explicit loop/model rebuild before WT-baseline physics.
- 5VEX contains a BRIL/T4 lysozyme-family fusion segment aligned to
  non-target UniProt `P00720` at auth residues 1002-1161; apo-priming
  strips it together with NNC0640 (`97V`).
- 6X1A is a PF-06882961 (`UK4`) GLP-1R:Gs active complex, not a GLP-1
  peptide-bound structure; apo-priming strips UK4, G proteins, and
  nanobody/partner chains.
