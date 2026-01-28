//! PRISM4D Correlation and Evidence-Pack Generator
//!
//! Production-grade site-level correlation and decision-artifact generator
//! for the NHS Cryo-UV cryptic site detection pipeline.
//!
//! # Features
//!
//! - Ranked candidate sites with confidence scores
//! - Tier 1 correlation (holo ligand proximity)
//! - Tier 2 correlation (truth residue precision/recall)
//! - Ablation analysis (baseline vs cryo vs cryo+UV)
//! - Decision artifacts (figures, 3D sessions, HTML/PDF reports)
//!
//! # CLI Contract
//!
//! ```bash
//! prism4d run --pdb APO.pdb --mode cryo-uv --replicates 5 \
//!     --wavelengths 258,274,280 --out results/ID \
//!     [--holo HOLO.pdb] [--truth-residues truth.json]
//! ```

pub mod config;
pub mod correlation;
pub mod inputs;
pub mod outputs;
pub mod pipeline;
pub mod sites;
pub mod ablation;
pub mod figures;
pub mod sessions;
pub mod reports;
pub mod event_cloud;
pub mod voxelize;
pub mod alignment;
pub mod finalize;
pub mod site_geometry;
pub mod site_metrics;

// Re-exports
pub use config::{ReportConfig, AblationConfig};
// Tier1/Tier2 correlation REMOVED per user requirement
pub use inputs::{CryoProbeResults};
pub use outputs::{OutputContract, SiteOutput};
pub use pipeline::{ReportPipeline, PipelineResult};
pub use sites::{CrypticSite, SiteMetrics, SiteRanking};
pub use ablation::{AblationResults, AblationMode};
pub use reports::{HtmlReport, PdfReport};
pub use outputs::SummaryJson;
pub use event_cloud::{EventCloud, PocketEvent, AblationPhase, TempPhase, EventWriter, read_events};
pub use voxelize::{VoxelGrid, Voxelizer, VoxelizationResult, write_mrc, voxelize_event_cloud};
pub use alignment::{Alignment, VoxelLigandOverlap, kabsch_align, align_structures, compute_voxel_ligand_overlap};
pub use finalize::{FinalizeStage, FinalizeResult};
// New site_metrics exports
pub use site_metrics::{
    TopologyData as MetricsTopologyData,
    SiteMetricsComputer,
    validate_coordinate_frames,
    sort_sites_deterministic,
};

/// Crate version (from Cargo.toml)
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Release version label for outputs (decoupled from crate version)
pub const PRISM4D_RELEASE: &str = "1.2.0-cryo-uv";

/// Check if PyMOL is installed and return path
pub fn find_pymol() -> Option<std::path::PathBuf> {
    which::which("pymol").ok()
}

/// Check if ChimeraX is installed and return path
pub fn find_chimerax() -> Option<std::path::PathBuf> {
    // Try common names
    for name in &["chimerax", "ChimeraX", "chimera"] {
        if let Ok(path) = which::which(name) {
            return Some(path);
        }
    }
    // Try common installation paths
    let common_paths = [
        "/usr/bin/chimerax",
        "/usr/local/bin/chimerax",
        "/opt/UCSF/ChimeraX/bin/ChimeraX",
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
    ];
    for p in common_paths {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

/// Check if wkhtmltopdf or playwright is installed for PDF generation
pub fn find_pdf_renderer() -> Option<(String, std::path::PathBuf)> {
    if let Ok(path) = which::which("wkhtmltopdf") {
        return Some(("wkhtmltopdf".to_string(), path));
    }
    if let Ok(path) = which::which("playwright") {
        return Some(("playwright".to_string(), path));
    }
    // Check for chromium/chrome for headless PDF
    for name in &["chromium", "chromium-browser", "google-chrome", "chrome"] {
        if let Ok(path) = which::which(name) {
            return Some((name.to_string(), path));
        }
    }
    None
}

/// Installation instructions for missing dependencies
pub fn dependency_install_instructions(dep: &str) -> String {
    match dep {
        "pymol" => r#"
PyMOL is required for .pse session generation.

Installation options:
  Ubuntu/Debian: sudo apt install pymol
  macOS:         brew install pymol
  Conda:         conda install -c conda-forge pymol-open-source

Or download from: https://pymol.org/
"#.to_string(),
        "chimerax" => r#"
UCSF ChimeraX is required for .cxs session generation.

Installation:
  Download from: https://www.cgl.ucsf.edu/chimerax/download.html

Ubuntu/Debian: Download .deb and: sudo dpkg -i chimerax*.deb
macOS:         Download .dmg and drag to Applications
"#.to_string(),
        "pdf" => r#"
A PDF renderer is required for report.pdf generation.

Options (install any one):
  wkhtmltopdf:  sudo apt install wkhtmltopdf  # Recommended
  playwright:   npm install -g playwright && playwright install
  chromium:     sudo apt install chromium-browser
"#.to_string(),
        _ => format!("Unknown dependency: {}", dep),
    }
}
