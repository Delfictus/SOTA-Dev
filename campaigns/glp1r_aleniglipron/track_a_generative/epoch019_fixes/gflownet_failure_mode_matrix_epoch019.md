# GFlowNet v1 — Failure-Mode Audit

Generated: 2026-05-25T09:37:39Z
Counts: PASS=7  WARN=2  FAIL=0  INFO=3

## mode_collapse
- **status:** PASS
- **evidence:** top anchor share = 0.04%
- **next action:** —

## cryptic_only_reward_hack
- **status:** PASS
- **evidence:** cryptic-only candidate share = 0.00%
- **next action:** —

## tiny_fragment_exploit
- **status:** PASS
- **evidence:** <12 heavy-atom share among filtered = 0.00%
- **next action:** —

## policy_reward_correlation
- **status:** WARN
- **evidence:** Spearman rho(logprob, reward) = -0.014
- **next action:** verify policy is targeting reward — check training curve

## pose_sensitivity
- **status:** INFO
- **evidence:** Rust oracle_scorer is SMILES-keyed lookup — pose_sensitivity = 0 by construction
- **next action:** to measure pose variance, rescore with physics-based oracle (out of v1 scope)

## training_set_memorization
- **status:** INFO
- **evidence:** top-100 ∩ survivors = 0.0% (expected ~100% — policy samples from anchor-resolved survivor SMILES)
- **next action:** for de novo generation, swap action space from anchor→survivor lookup to atom-level construction

## top100_duplicate_collapse
- **status:** PASS
- **evidence:** top-100 unique SMILES = 100/100
- **next action:** —

## action_family_imbalance_top100
- **status:** PASS
- **evidence:** max anchor share in top-100 = 1.0%
- **next action:** —

## production_medchem_triage
- **status:** PASS
- **evidence:** PAINS pass=100.0%, BRENK pass=86.0%, oral pass=100.0%
- **next action:** —

## biased_agonism_verification
- **status:** WARN
- **evidence:** biased_agonism_confirmed_top50=37/50; top100=48/100 at pi_clash_lock>0.5; lock-specific pi_clash field is present
- **next action:** rescore top-50 with lock-specific Rust oracle channel before dossier promotion

## invalid_chemistry_rate
- **status:** PASS
- **evidence:** invalid-status share = 0.00%
- **next action:** —

## uncertainty_concentration
- **status:** INFO
- **evidence:** median reward_cv = 0.000 (0 expected with deterministic backend)
- **next action:** non-deterministic backend (physics-based) required to surface real uncertainty
