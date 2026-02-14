//! Validation, Metrics, and Explainability Module
//!
//! Comprehensive validation toolkit for PRISM-LBS supporting publication-ready benchmarks:
//!
//! ## DCC/DCA Metrics
//! Industry-standard binding site prediction metrics:
//! - DCC (Distance to Closest Contact): Min distance from pocket to ligand
//! - DCA (Distance to Center of Active site): Centroid-to-centroid distance
//! - Top-N Success Rates: Success with best N pockets
//!
//! ## DVO (Discretized Volume Overlap)
//! Spatial overlap metrics:
//! - Jaccard index for volume overlap
//! - Dice coefficient
//! - Grid-based discretization
//!
//! ## Residue-Level Metrics
//! For CryptoBench and residue-centric evaluation:
//! - AUC (Area Under ROC Curve)
//! - AUPRC (Area Under Precision-Recall Curve)
//! - MCC (Matthews Correlation Coefficient)
//! - F1 Score
//!
//! ## Pocket-Level Metrics
//! For LIGYSIS and standard benchmarks:
//! - Top-N+2 recall (LIGYSIS standard)
//! - Recall at various rank thresholds
//!
//! ## Statistical Tests
//! For method comparison:
//! - McNemar's test for paired comparisons
//! - Bootstrap confidence intervals
//! - Wilcoxon signed-rank test
//!
//! ## Ligand Parsing
//! Multi-format ligand coordinate extraction:
//! - PDB HETATM records
//! - SDF/MOL files
//! - MOL2 (Tripos format)
//! - XYZ coordinate files
//!
//! ## Benchmark Runner
//! Automated validation against standard datasets:
//! - PDBBind refined set
//! - CryptoSite/CryptoBench
//! - LIGYSIS
//! - ASBench
//!
//! ## Explainability
//! Interpretable AI for drug discovery:
//! - Per-residue contribution scores
//! - Druggability factor decomposition
//! - Confidence breakdown
//! - Human-readable reasoning
//!
//! ## Docking Integration
//! Seamless docking workflow support:
//! - AutoDock Vina box generation
//! - Pharmacophore feature extraction
//! - PyMOL visualization scripts

pub mod benchmark;
pub mod docking;
pub mod dvo;
pub mod explainability;
pub mod ligand_parser;
pub mod metrics;
pub mod pocket_metrics;
pub mod residue_metrics;
pub mod statistical_tests;

// Re-export key types
pub use metrics::{
    BenchmarkCase, BenchmarkSummary, CaseResult, TopNMetrics, ValidationMetrics,
    DEFAULT_SUCCESS_THRESHOLD,
};

pub use dvo::{
    calculate_dvo, calculate_dvo_from_atoms, calculate_dvo_from_residues, calculate_dvo_simple,
    DvoResult, DVO_EXCELLENT, DVO_GOOD, DVO_SUCCESS,
};

pub use residue_metrics::{
    calculate_all_residue_metrics, calculate_auc, calculate_auprc, calculate_f1, calculate_mcc,
    calculate_pr_curve, calculate_roc_curve, find_optimal_f1_threshold, find_optimal_mcc_threshold,
    predictions_from_residue_lists, ConfusionMatrix, ResidueMetrics, ResiduePrediction,
};

pub use pocket_metrics::{
    calculate_recall_at_rank, calculate_recall_at_ranks, calculate_top_n_plus_2_recall,
    calculate_top_n_recall, match_predictions_to_ground_truth, AggregatedPocketMetrics,
    GroundTruthSite, PocketMatch, PocketPrediction, RecallAtRanks, DCC_LIGYSIS, DCC_RELAXED,
    DCC_STRICT,
};

pub use statistical_tests::{
    bootstrap_ci, bootstrap_ci_binary, mcnemar_test, paired_t_test, wilcoxon_signed_rank,
    ConfidenceInterval, McNemarResult, WilcoxonResult,
};

pub use ligand_parser::{Ligand, LigandAtom, LigandParser};

pub use benchmark::{
    validate_single, BenchmarkComparison, BenchmarkReport, CaseEvaluation, PDBBindBenchmark,
};

pub use explainability::{
    AssessmentStatus, ConfidenceBreakdown, DetectionSignal, DruggabilityClass,
    DruggabilityFactors, ExplainabilityEngine, FactorAssessment, PocketExplanation,
    ResidueContribution, ResidueFactors, ResidueRole,
};

pub use docking::{
    DockingSite, DockingSiteGenerator, PharmacophoreFeature, PharmacophoreModel,
    PharmacophoreType, VinaBox,
};
