//! World-Class Allosteric Detection Module for PRISM-LBS
//!
//! A 4-stage pipeline for Nature-publication-quality allosteric site detection:
//!
//! - Stage 1: Structural Analysis (Domain Decomposition + Hinge Detection)
//! - Stage 2: Evolutionary Signal (MSA Conservation + Sequence Analysis)
//! - Stage 3: Network Analysis (Residue Interaction Network + Allosteric Coupling)
//! - Stage 4: Hybrid Consensus (Multi-module Scoring + Backtrack Gap Analysis)
//!
//! ## Performance Targets
//! - Classic binding sites: 94% detection
//! - Cryptic sites: 82% detection
//! - Allosteric sites: 80% detection
//! - Overall: 85%+ detection rate
//!
//! Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.

pub mod types;
pub mod domain_decomposition;
pub mod gpu_apsp;
pub mod hinge_detection;
pub mod msa_conservation;
pub mod residue_network;
pub mod allosteric_coupling;
pub mod consensus_engine;
pub mod backtrack;
pub mod detector;

// Re-exports
pub use types::*;
pub use domain_decomposition::DomainDecomposer;
pub use gpu_apsp::{GpuFloydWarshall, betweenness_centrality, closeness_centrality};
pub use hinge_detection::HingeDetector;
pub use msa_conservation::ConservationAnalyzer;
pub use residue_network::ResidueNetworkAnalyzer;
pub use allosteric_coupling::AllostericCouplingAnalyzer;
pub use consensus_engine::HybridConsensusEngine;
pub use backtrack::BacktrackAnalyzer;
pub use detector::AllostericDetector;
