Read PRISM4D_BLEEDING_EDGE_PLAN.md and execute PHASE 1 only.

CRITICAL CONSTRAINTS:
- Do NOT modify any source code in crates/
- Do NOT run cargo build
- Do NOT touch the kernel, PTX, or binary
- This is PURE ANALYSIS on existing data in /mnt/storage/prism-outputs/runs/v1.1-physics/

Step 1A: Install ijson for streaming JSON parsing:
  pip install ijson polars pyarrow --break-system-packages

Step 1B: Build the per-site spike feature extractor.

For EACH of the 18 targets in /mnt/storage/prism-outputs/runs/v1.1-physics/:
  - Read binding_sites.json for site metadata (site_id, therm_class, hysteresis_asymmetry, ccns_tau, quality_score, spike_count, estimated_volume, druggability_score)
  - For each site, find its matching spike_events.json file by site_id
  - Stream the spike file with ijson. Do NOT use json.load(). Sample strategy: read spikes in 3 passes using file byte offsets — first 5MB (cold_hold spikes), middle 5MB (heating/warm spikes), last 5MB (cooling/return spikes). This ensures all 5 phases are represented despite timestep-sorted files.
  - Compute per-site: source_entropy, intensity_entropy (100 bins), n_unique_sources, n_unique_intensities, efp_fraction, lif_fraction, phase_coverage (fraction of 5 phases with >1% spikes), cold_hot_ratio, phase_entropy, heating_cooling_ratio, burst_index (fraction of spikes in top 10% densest timestep bins), inter_spike_cv (CV of inter-spike intervals)
  - If a spike file is not found for a site, compute channel features as NaN and use only binding_sites.json features

Output: /mnt/storage/prism-outputs/runs/v1.1-physics/site_features.csv with columns:
  target, site_id, rank_original, therm_class, source_entropy, intensity_entropy, n_unique_sources, n_unique_intensities, efp_fraction, lif_fraction, phase_coverage, cold_hot_ratio, phase_entropy, heating_cooling_ratio, burst_index, inter_spike_cv, hysteresis_asymmetry, ccns_tau, quality_score, spike_count, estimated_volume, druggability_score

Step 1C: Build THREE re-ranking methods:

Method 1 (Supervised): Logistic regression. Label = therm_class==CRYPTIC. Features = all spike entropy + binding site features. LOTO CV. Report AUROC, per-target predictions.

Method 2 (Unsupervised): Fisher discriminant ratio per feature. Weight features by Fisher ratio. Composite score = weighted sum. No training needed.

Method 3 (Information-theoretic): Mutual information I(feature; therm_class) per feature. Weight by MI. Composite score = weighted sum.

For each method, re-rank all sites per target. For each CRYPTIC site, report old rank vs new rank.

Step 1D: Produce ENTROPY_RANKING.md with:
  - Per-target table: Target | CRYPTIC site | Old Rank | New Rank (M1) | New Rank (M2) | New Rank (M3)
  - Summary: | Method | Mean CRYPTIC Rank | SR@1 | SR@3 | SR@5 | AUROC |
  - Which method is best and why

Step 1E: Feature importance analysis:
  - Top 5 features by weight for the best method
  - Feature correlation matrix
  - CRYPTIC vs non-CRYPTIC distribution per feature (save as feature_analysis/ PNGs)

Save everything to /mnt/storage/prism-outputs/runs/v1.1-physics/. Then STOP and show me the summary table from Step 1D before proceeding.
