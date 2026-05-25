# Failure-mode matrix

| failure mode | status | evidence |
|---|---|---|
| mode_collapse | PASS | top anchor share = 3.97% |
| cryptic_only_reward_hack | PASS | cryptic-only candidate share = 0.00% |
| tiny_fragment_exploit | PASS | <12 heavy-atom share among filtered = 0.00% |
| policy_reward_correlation | WARN | Spearman rho(logprob, reward) = -0.014 |
| pose_sensitivity | INFO | Rust oracle_scorer is SMILES-keyed lookup — pose_sensitivity = 0 by construction |
| training_set_memorization | INFO | top-100 ∩ survivors = 0.0% (expected ~100% — policy samples from anchor-resolved survivor SMILES) |
| top100_duplicate_collapse | PASS | top-100 unique SMILES = 100/100 |
| action_family_imbalance_top100 | PASS | max anchor share in top-100 = 1.0% |
| production_medchem_triage | PASS | PAINS pass=100.0%, BRENK pass=88.0%, oral pass=100.0% |
| biased_agonism_verification | PASS | biased_agonism_confirmed_top50=50/50; top100=100/100 at pi_clash_lock>0.5; lock-specific pi_clash field is present |
| invalid_chemistry_rate | PASS | invalid-status share = 0.00% |
| uncertainty_concentration | INFO | median reward_cv = 0.000 (0 expected with deterministic backend) |
