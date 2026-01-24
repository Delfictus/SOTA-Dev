# Reprocessed APO Structures

## Date: 2026-01-17

## Purpose
These 7 structures were reprocessed to fix high-energy issues caused by:
1. Bound ligands invalidating blind cryptic site detection
2. Missing/malformed AMBER parameters (angles, dihedrals)

## Structures Reprocessed

| PDB | Protein | Pathogen | Atoms | Ligands Removed |
|-----|---------|----------|-------|-----------------|
| 1HXY | gp120 Envelope | HIV-1 | 9,444 | ZN |
| 2VWD | Attachment G | Nipah | 12,926 | NAG, GBL, CL |
| 3SQQ | VP35 | Marburg | 4,705 | 99Z, EDO |
| 4B7Q | Neuraminidase | Influenza H7N9 | 23,312 | Glycans (A2xxx, B2xxx, etc.) |
| 5IRE | Envelope | Zika | 26,297 | (chains only) |
| 6LU7 | Main Protease | SARS-CoV-2 | 4,730 | 02J, PJE, 010 |
| 6M0J | RBD + ACE2 | SARS-CoV-2 | 12,510 | NAG, CL, ZN |

## Processing Pipeline

1. **Strip HETATM** → APO structure (protein only)
2. **Stage 1: Sanitize** → Add hydrogens, fix missing atoms
3. **Stage 2: Topology** → Full AMBER ff14SB parameters + minimization

## Validation

All structures now have:
- ✓ Proper bond parameters (k, r0)
- ✓ Angle parameters (k, θ0)
- ✓ Dihedral parameters (periodicity, phase)
- ✓ H-constraint clusters
- ✓ Masses, charges, LJ parameters
- ✓ Normal potential energies (expected ~1e5-1e6 kcal/mol)

## File Locations

- Raw APO: `data/reprocessed_apo/raw/`
- Sanitized: `data/reprocessed_apo/sanitized/`
- Topologies: `data/reprocessed_apo/topologies/`
