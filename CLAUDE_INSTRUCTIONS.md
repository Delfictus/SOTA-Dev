  # CLAUDE DEVELOPMENT INSTRUCTIONS

  ## CRITICAL: Directory Rules

  | Directory | Purpose | Claude Can Modify? |
  |-----------|---------|-------------------|
  | ~/Desktop/PRISM4D-v1.1.0-STABLE | Validated release | NO NEVER |
  | ~/Desktop/PRISM4D-dev | Active development | YES |

  ## Current Branch: feature/explicit-solvent

  ## What's Working (DO NOT BREAK):
  - Implicit solvent MD engine (1ns validated)
  - RMSD/RMSF analysis pipeline
  - Publication figure generation
  - Docker containerization (CUDA 13.0)

  ## Development Goals:
  1. Add explicit solvent (TIP3P water model)
  2. Implement PME electrostatics
  3. Remove heavy atom restraints
  4. Target: 100 ns production runs

  ## Before Any Changes:
  ```bash
  pwd         # Must be PRISM4D-dev
  git branch  # Must be feature/explicit-solvent
