//! Soft-spot detection module for cryptic binding sites
//!
//! This module implements detection of cryptic (hidden) binding sites that are not
//! currently open cavities but have the biophysical characteristics indicating they
//! could become druggable pockets upon conformational change.
//!
//! ## Scientific Basis
//!
//! Even when closed, cryptic sites exhibit measurable signatures:
//! - **Elevated B-factors**: Conformational flexibility
//! - **Low packing density**: Room to open
//! - **Hydrophobic clustering**: Energetically favorable for ligand binding
//!
//! ## Enhanced Detection (v2)
//!
//! Beyond B-factor flexibility, we now incorporate:
//! - **Normal Mode Analysis (NMA)**: Collective motions from ANM
//! - **Contact Order Analysis**: Low contact order = easier conformational change
//! - **Evolutionary Conservation**: Conserved residues without structural role
//! - **Probe Clustering**: FTMap-style computational fragment mapping
//!
//! ## Usage
//!
//! ```ignore
//! use prism_lbs::softspot::{SoftSpotDetector, EnhancedSoftSpotDetector};
//!
//! // Classic B-factor-based detection
//! let detector = SoftSpotDetector::new();
//! let candidates = detector.detect(&atoms);
//!
//! // Enhanced multi-signal detection
//! let enhanced = EnhancedSoftSpotDetector::new();
//! let candidates = enhanced.detect(&atoms);
//! ```
//!
//! ## References
//!
//! - Cimermancic et al. (2016) - CryptoSite
//! - Kozakov et al. (2015) - FTMap
//! - Vajda et al. (2018) - Cryptic sites at domain interfaces
//! - Bahar et al. (1997) - Gaussian Network Model
//! - Plaxco et al. (1998) - Contact order and protein folding
//! - Capra & Singh (2007) - Conservation analysis

mod constants;
mod detector;
mod types;

// Enhanced detection modules (v2)
pub mod conservation;
pub mod contact_order;
pub mod enhanced;
pub mod gpu_lanczos;
pub mod lanczos;
pub mod nma;
pub mod probe_clustering;

pub use constants::*;
pub use detector::{SoftSpotConfig, SoftSpotDetector};
pub use enhanced::{EnhancedSoftSpotConfig, EnhancedSoftSpotDetector};
pub use types::{CrypticCandidate, CrypticConfidence, FlexibleResidue};

// Re-export enhancement analyzers for advanced use
pub use conservation::{ConservationAnalyzer, ConservationResult};
pub use contact_order::{ContactOrderAnalyzer, ContactOrderResult};
pub use gpu_lanczos::{GpuEigenResult, GpuLanczosEigensolver};
pub use lanczos::{EigenResult, LanczosEigensolver};
pub use nma::{NmaAnalyzer, NmaResult};
pub use probe_clustering::{ProbeClusteringAnalyzer, ProbeClusteringResult};
