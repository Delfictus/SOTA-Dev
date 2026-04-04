# PRISM-TWIN Determinism Invariants

## Rule 1: Standard mode unchanged
When `--coupled-twin` is NOT specified, the engine MUST produce output
identical to the pre-PRISM-TWIN codebase. Zero regression. The coupled_md
code path is not entered.

## Rule 2: Deterministic channels preserved in coupled mode
When `--coupled-twin` IS specified with exchange DISABLED (Gate 1 config):
- **UV channel**: BIT-EXACT with single-stream baseline (same seed, same photon absorption)
- **RAF channel**: BIT-EXACT within ±1 count (FP accumulation order)
- These channels depend only on atomic positions and aromatic classification,
  which are computed identically regardless of stepping pattern.

## Rule 3: Stochastic channels may diverge under interleaving
- **EFP channel**: EXPECTED to diverge (0.1–1.0% per 10K steps)
- **LADD channel**: EXPECTED to diverge (same reason)
- The Langevin thermostat draws random forces every step. When stepping
  is interleaved (A-B-A-B), the CUDA RNG state consumption order differs
  from single-stream (A-A-A-A). Different random forces → different velocities
  → different EFP/LADD detection.
- This is equivalent to running with a different seed: the DISTRIBUTION
  is statistically identical, the TRAJECTORY differs.
- Divergence GROWS with time (chaotic system). This is expected.

## Rule 4: Exchange coupling is observation-only
When exchange IS enabled (Gate 2+):
- Forces in each stream remain INDEPENDENT
- Only oscillator thresholds are modified by the other stream's spike density
- The Hamiltonian is unchanged — thermodynamic quantities remain valid per-stream
- Cross-stream coupling affects WHAT the detectors notice, not WHAT the atoms do

## Rule 5: How to verify
- Run standard mode, record UV/RAF/EFP counts at steps 10K/20K/30K
- Run coupled mode stream A with same seed, exchange disabled
- UV counts MUST match exactly
- RAF counts MUST match within ±1 per checkpoint
- EFP counts WILL differ — this is correct behavior, not a bug
- If UV diverges: STOP. Force field is corrupted. DO NOT proceed.
