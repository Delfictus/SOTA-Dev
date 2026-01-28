# PRISM Phase 6: Implementation Plan - Part 2
## Weeks 3-4: PRISM-NOVA Integration & Weeks 5-8 Overview

---

## 5. Weeks 3-4: PRISM-NOVA Integration

### Task 3.2: NOVA Cryptic Sampler (Continued)

The complete `NovaCrypticSampler` implementation is in Part 1. Key methods:

- `new(context)` - Initialize with CUDA context (NO CPU fallback)
- `load_structure(pdb_content)` - Load and sanitize PDB
- `sample()` - Run HMC sampling with TDA tracking
- Returns `NovaSamplingResult` with conformations, Betti history, quality metrics

---

### Task 3.3: Create Apo-Holo Benchmark

**File**: `crates/prism-validation/src/apo_holo_benchmark.rs`

This is the "killer demo" - proving PRISM can predict conformational changes.

```rust
//! Apo-Holo Conformational Change Benchmark
//!
//! Tests PRISM's ability to predict conformational changes:
//! 1. Start from apo (ligand-free) structure
//! 2. Sample with NOVA
//! 3. Check if any conformation approaches holo (ligand-bound) state

use anyhow::{Result, Context};
use std::sync::Arc;
use cudarc::driver::CudaContext;

use crate::nova_cryptic_sampler::{NovaCrypticSampler, NovaCrypticConfig};
use crate::pdb_sanitizer::PdbSanitizer;

/// Classic apo-holo pairs with known conformational changes
pub const APO_HOLO_PAIRS: &[ApoHoloPair] = &[
    ApoHoloPair { apo: "1AKE", holo: "4AKE", name: "Adenylate kinase", motion: MotionType::DomainClosure },
    ApoHoloPair { apo: "2LAO", holo: "1LST", name: "Lysine-binding protein", motion: MotionType::HingeMotion },
    ApoHoloPair { apo: "1GGG", holo: "1WDN", name: "Calmodulin", motion: MotionType::DomainRotation },
    ApoHoloPair { apo: "1OMP", holo: "1ANF", name: "Maltose-binding protein", motion: MotionType::DomainClosure },
    ApoHoloPair { apo: "1RX2", holo: "1RX4", name: "Ribonuclease", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "3CHY", holo: "2CHE", name: "CheY", motion: MotionType::SmallRotation },
    ApoHoloPair { apo: "1EX6", holo: "1EX7", name: "Galectin", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "1STP", holo: "1SWB", name: "Streptavidin", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "1AJJ", holo: "1AJK", name: "Guanylate kinase", motion: MotionType::DomainClosure },
    ApoHoloPair { apo: "1PHP", holo: "1PHN", name: "Phosphotransferase", motion: MotionType::HingeMotion },
    ApoHoloPair { apo: "1BTL", holo: "1BTM", name: "Beta-lactamase", motion: MotionType::SmallRotation },
    ApoHoloPair { apo: "2CPL", holo: "1CWA", name: "Cyclophilin", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "1BMD", holo: "1BMC", name: "Biotin-binding", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "1URN", holo: "1URP", name: "Ubiquitin", motion: MotionType::SmallRotation },
    ApoHoloPair { apo: "1HOE", holo: "1HOF", name: "Alpha-amylase inhibitor", motion: MotionType::LoopMotion },
];

#[derive(Debug, Clone, Copy)]
pub struct ApoHoloPair {
    pub apo: &'static str,
    pub holo: &'static str,
    pub name: &'static str,
    pub motion: MotionType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MotionType {
    DomainClosure,    // Large hinge-bending (>5Å)
    DomainRotation,   // Rigid body rotation
    HingeMotion,      // Classic hinge movement
    LoopMotion,       // Flexible loop rearrangement
    SmallRotation,    // Minor conformational shifts (<2Å)
}

impl MotionType {
    /// Success threshold (min RMSD to holo) for this motion type
    pub fn success_threshold(&self) -> f32 {
        match self {
            MotionType::SmallRotation => 1.5,
            MotionType::LoopMotion => 2.0,
            MotionType::HingeMotion => 2.5,
            MotionType::DomainRotation => 3.0,
            MotionType::DomainClosure => 3.5,
        }
    }
}

/// Result for single apo-holo validation
#[derive(Debug, Clone)]
pub struct ApoHoloResult {
    pub apo_pdb: String,
    pub holo_pdb: String,
    pub name: String,
    pub motion_type: MotionType,
    
    /// Starting RMSD (apo vs holo)
    pub apo_holo_rmsd: f32,
    
    /// Best (minimum) RMSD to holo achieved
    pub min_rmsd_to_holo: f32,
    
    /// Sample index that achieved best RMSD
    pub best_sample_idx: usize,
    
    /// Improvement: apo_holo_rmsd - min_rmsd_to_holo
    pub rmsd_improvement: f32,
    
    /// Did we approach holo state? (min_rmsd < threshold)
    pub success: bool,
    
    /// Sample index of first significant improvement
    pub time_to_open: Option<usize>,
    
    /// All RMSD values to holo
    pub rmsd_trajectory: Vec<f32>,
}

/// Benchmark runner
pub struct ApoHoloBenchmark {
    context: Arc<CudaContext>,
    config: NovaCrypticConfig,
    data_dir: String,
    results: Vec<ApoHoloResult>,
}

impl ApoHoloBenchmark {
    pub fn new(context: Arc<CudaContext>, data_dir: &str) -> Self {
        Self {
            context,
            config: NovaCrypticConfig::default(),
            data_dir: data_dir.to_string(),
            results: Vec::new(),
        }
    }
    
    pub fn with_config(mut self, config: NovaCrypticConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Run benchmark on all pairs
    pub fn run_all(&mut self) -> Result<ApoHoloBenchmarkSummary> {
        log::info!("Starting apo-holo benchmark on {} pairs", APO_HOLO_PAIRS.len());
        
        for pair in APO_HOLO_PAIRS {
            match self.run_pair(pair) {
                Ok(result) => {
                    log::info!("  {} {}: {:.2}Å → {:.2}Å ({})",
                               if result.success { "✓" } else { "✗" },
                               pair.name,
                               result.apo_holo_rmsd,
                               result.min_rmsd_to_holo,
                               if result.success { "SUCCESS" } else { "FAILED" });
                    self.results.push(result);
                }
                Err(e) => {
                    log::error!("  ✗ {} FAILED: {}", pair.name, e);
                }
            }
        }
        
        Ok(self.summarize())
    }
    
    /// Run benchmark on single pair
    pub fn run_pair(&self, pair: &ApoHoloPair) -> Result<ApoHoloResult> {
        // Load PDB files
        let apo_path = format!("{}/{}_apo.pdb", self.data_dir, pair.apo);
        let holo_path = format!("{}/{}_holo.pdb", self.data_dir, pair.holo);
        
        let apo_content = std::fs::read_to_string(&apo_path)
            .context(format!("Failed to read {}", apo_path))?;
        let holo_content = std::fs::read_to_string(&holo_path)
            .context(format!("Failed to read {}", holo_path))?;
        
        // Sanitize structures
        let sanitizer = PdbSanitizer::new();
        let apo_struct = sanitizer.sanitize(&apo_content)?;
        let holo_struct = sanitizer.sanitize(&holo_content)?;
        
        // Get Cα coordinates
        let apo_ca = apo_struct.get_ca_coords();
        let holo_ca = holo_struct.get_ca_coords();
        
        // Align lengths
        let n = apo_ca.len().min(holo_ca.len());
        let apo_ca: Vec<_> = apo_ca.into_iter().take(n).collect();
        let holo_ca: Vec<_> = holo_ca.into_iter().take(n).collect();
        
        // Baseline RMSD
        let apo_holo_rmsd = compute_rmsd(&apo_ca, &holo_ca);
        
        // Run NOVA sampling from apo
        let mut sampler = NovaCrypticSampler::new(Arc::clone(&self.context))?
            .with_config(self.config.clone());
        sampler.load_structure(&apo_content)?;
        
        let sampling_result = sampler.sample()?;
        
        // Compute RMSD to holo for each sample
        let rmsd_trajectory: Vec<f32> = sampling_result.conformations.iter()
            .map(|conf| {
                let trimmed: Vec<_> = conf.iter().take(n).cloned().collect();
                compute_rmsd(&trimmed, &holo_ca)
            })
            .collect();
        
        // Find best
        let (best_idx, &min_rmsd) = rmsd_trajectory.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let rmsd_improvement = apo_holo_rmsd - min_rmsd;
        let success = min_rmsd < pair.motion.success_threshold();
        
        // Find time to first significant opening (30% improvement)
        let opening_threshold = apo_holo_rmsd * 0.7;
        let time_to_open = rmsd_trajectory.iter()
            .position(|&r| r < opening_threshold);
        
        Ok(ApoHoloResult {
            apo_pdb: pair.apo.to_string(),
            holo_pdb: pair.holo.to_string(),
            name: pair.name.to_string(),
            motion_type: pair.motion,
            apo_holo_rmsd,
            min_rmsd_to_holo: min_rmsd,
            best_sample_idx: best_idx,
            rmsd_improvement,
            success,
            time_to_open,
            rmsd_trajectory,
        })
    }
    
    /// Generate summary
    pub fn summarize(&self) -> ApoHoloBenchmarkSummary {
        let n_total = self.results.len();
        let n_success = self.results.iter().filter(|r| r.success).count();
        
        let mean_improvement = if n_total > 0 {
            self.results.iter().map(|r| r.rmsd_improvement).sum::<f32>() / n_total as f32
        } else { 0.0 };
        
        let mean_min_rmsd = if n_total > 0 {
            self.results.iter().map(|r| r.min_rmsd_to_holo).sum::<f32>() / n_total as f32
        } else { 0.0 };
        
        ApoHoloBenchmarkSummary {
            n_total,
            n_success,
            success_rate: n_success as f32 / n_total.max(1) as f32,
            mean_rmsd_improvement: mean_improvement,
            mean_min_rmsd_to_holo: mean_min_rmsd,
            results: self.results.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApoHoloBenchmarkSummary {
    pub n_total: usize,
    pub n_success: usize,
    pub success_rate: f32,
    pub mean_rmsd_improvement: f32,
    pub mean_min_rmsd_to_holo: f32,
    pub results: Vec<ApoHoloResult>,
}

impl ApoHoloBenchmarkSummary {
    /// Generate markdown report
    pub fn to_report(&self) -> String {
        let mut s = String::new();
        s.push_str("# Apo-Holo Benchmark Results\n\n");
        s.push_str(&format!("**Success Rate**: {}/{} ({:.0}%)\n\n", 
                            self.n_success, self.n_total, self.success_rate * 100.0));
        s.push_str(&format!("**Mean Improvement**: {:.2} Å\n", self.mean_rmsd_improvement));
        s.push_str(&format!("**Mean Min RMSD**: {:.2} Å\n\n", self.mean_min_rmsd_to_holo));
        
        s.push_str("| Protein | Apo→Holo | Best | Δ | Status |\n");
        s.push_str("|---------|----------|------|---|--------|\n");
        for r in &self.results {
            s.push_str(&format!("| {} | {:.2}Å | {:.2}Å | {:.2}Å | {} |\n",
                                r.name, r.apo_holo_rmsd, r.min_rmsd_to_holo,
                                r.rmsd_improvement,
                                if r.success { "✓" } else { "✗" }));
        }
        s
    }
    
    /// Generate LaTeX table
    pub fn to_latex(&self) -> String {
        let mut s = String::new();
        s.push_str("\\begin{table}[h]\n\\centering\n");
        s.push_str("\\caption{Apo-Holo Conformational Change Prediction}\n");
        s.push_str("\\begin{tabular}{lcccc}\n\\toprule\n");
        s.push_str("Protein & Apo$\\to$Holo & Min RMSD & $\\Delta$ & Success \\\\\n");
        s.push_str("\\midrule\n");
        
        for r in &self.results {
            s.push_str(&format!("{} & {:.2}\\AA & {:.2}\\AA & {:.2}\\AA & {} \\\\\n",
                                r.name.replace("_", "\\_"),
                                r.apo_holo_rmsd, r.min_rmsd_to_holo, r.rmsd_improvement,
                                if r.success { "\\checkmark" } else { "$\\times$" }));
        }
        
        s.push_str("\\midrule\n");
        s.push_str(&format!("\\textbf{{Total}} & & {:.2}\\AA & {:.2}\\AA & {:.0}\\% \\\\\n",
                            self.mean_min_rmsd_to_holo, self.mean_rmsd_improvement,
                            self.success_rate * 100.0));
        s.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        s
    }
}

/// Compute RMSD between two conformations
fn compute_rmsd(conf1: &[[f32; 3]], conf2: &[[f32; 3]]) -> f32 {
    assert_eq!(conf1.len(), conf2.len());
    let n = conf1.len() as f32;
    let sum_sq: f32 = conf1.iter().zip(conf2.iter())
        .map(|(a, b)| {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            dx*dx + dy*dy + dz*dz
        })
        .sum();
    (sum_sq / n).sqrt()
}
```

---

### Week 3-4 Checklist

```
□ pdb_sanitizer.rs compiles and tests pass
□ nova_cryptic_sampler.rs compiles
□ apo_holo_benchmark.rs compiles
□ Test on small PDB (dipeptide) passes
□ Test on 3CSY (Ebola GP trimer) passes
□ Interface residue detection works for multi-chain
□ TDA Betti-2 tracking shows void formation
□ Acceptance rate > 20% (adjust temperature if needed)
□ Ensemble quality metrics computed
□ Apo-holo benchmark runs on 1AKE/4AKE pair
□ At least 10/15 apo-holo pairs show improvement
```

---

## 6. Weeks 5-6: CryptoBench & Ablation Study

### Objective

- Full validation on 1107 CryptoBench structures
- Ablation study proving component contributions
- Failure case analysis

### Files to Create

1. `crates/prism-validation/src/cryptobench_dataset.rs`
2. `crates/prism-validation/src/cryptobench_benchmark.rs`
3. `crates/prism-validation/src/ablation.rs`
4. `crates/prism-validation/src/failure_analysis.rs`

---

### Task 5.1: CryptoBench Dataset Loader

**File**: `crates/prism-validation/src/cryptobench_dataset.rs`

```rust
//! CryptoBench dataset loader
//! 
//! Handles the 1107-structure benchmark with 885/222 train/test split.

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoBenchEntry {
    pub structure_id: String,
    pub pdb_path: String,
    pub chain_id: String,
    pub binding_residues: Vec<i32>,
    pub pocket_type: Option<String>,
    pub ligand_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoBenchManifest {
    pub name: String,
    pub version: String,
    pub train_entries: Vec<CryptoBenchEntry>,
    pub test_entries: Vec<CryptoBenchEntry>,
}

pub struct CryptoBenchDataset {
    pub manifest: CryptoBenchManifest,
    pub base_path: String,
}

impl CryptoBenchDataset {
    /// Load dataset from manifest file
    pub fn load(manifest_path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(manifest_path)
            .context("Failed to read manifest")?;
        let manifest: CryptoBenchManifest = serde_json::from_str(&content)
            .context("Failed to parse manifest JSON")?;
        
        let base_path = Path::new(manifest_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        
        log::info!("Loaded CryptoBench: {} train, {} test structures",
                   manifest.train_entries.len(), manifest.test_entries.len());
        
        Ok(Self { manifest, base_path })
    }
    
    /// Get training entries
    pub fn train_entries(&self) -> &[CryptoBenchEntry] {
        &self.manifest.train_entries
    }
    
    /// Get test entries
    pub fn test_entries(&self) -> &[CryptoBenchEntry] {
        &self.manifest.test_entries
    }
    
    /// Get ground truth binding residues as set
    pub fn ground_truth(&self, structure_id: &str) -> Option<HashSet<i32>> {
        self.manifest.train_entries.iter()
            .chain(self.manifest.test_entries.iter())
            .find(|e| e.structure_id == structure_id)
            .map(|e| e.binding_residues.iter().cloned().collect())
    }
    
    /// Load PDB content for entry
    pub fn load_pdb(&self, entry: &CryptoBenchEntry) -> Result<String> {
        let path = format!("{}/{}", self.base_path, entry.pdb_path);
        std::fs::read_to_string(&path)
            .context(format!("Failed to load PDB: {}", path))
    }
    
    /// Validate dataset integrity
    pub fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();
        
        for entry in self.manifest.train_entries.iter()
            .chain(self.manifest.test_entries.iter()) 
        {
            report.total += 1;
            
            let path = format!("{}/{}", self.base_path, entry.pdb_path);
            if !Path::new(&path).exists() {
                report.missing_pdbs.push(entry.structure_id.clone());
                continue;
            }
            
            if entry.binding_residues.is_empty() {
                report.no_binding_site.push(entry.structure_id.clone());
                continue;
            }
            
            report.valid += 1;
        }
        
        Ok(report)
    }
}

#[derive(Debug, Default)]
pub struct ValidationReport {
    pub total: usize,
    pub valid: usize,
    pub missing_pdbs: Vec<String>,
    pub no_binding_site: Vec<String>,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        self.missing_pdbs.is_empty() && self.no_binding_site.is_empty()
    }
}
```

---

### Task 5.2: Ablation Study Framework

**File**: `crates/prism-validation/src/ablation.rs`

```rust
//! Ablation study framework
//!
//! Tests 6 variants to prove component contributions:
//! 1. ANM only (baseline)
//! 2. ANM + GPU-SNN
//! 3. NOVA only
//! 4. NOVA + CPU-SNN (for comparison only, not production)
//! 5. NOVA + GPU-SNN (no TDA)
//! 6. Full pipeline (NOVA + GPU-SNN + TDA + RLS)

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AblationVariant {
    /// Baseline: ANM sampling, no learning
    AnmOnly,
    
    /// ANM + GPU 512-neuron scorer
    AnmGpuSnn,
    
    /// NOVA sampling, basic scorer
    NovaOnly,
    
    /// NOVA + CPU 64-neuron (benchmark only, NOT production)
    NovaCpuSnn,
    
    /// NOVA + GPU 512-neuron, no TDA
    NovaGpuSnn,
    
    /// Full pipeline: NOVA + GPU-SNN + TDA + RLS
    Full,
}

impl AblationVariant {
    pub fn name(&self) -> &'static str {
        match self {
            Self::AnmOnly => "ANM Only",
            Self::AnmGpuSnn => "ANM + GPU-SNN",
            Self::NovaOnly => "NOVA Only",
            Self::NovaCpuSnn => "NOVA + CPU-SNN",
            Self::NovaGpuSnn => "NOVA + GPU-SNN",
            Self::Full => "Full Pipeline",
        }
    }
    
    pub fn uses_nova(&self) -> bool {
        matches!(self, Self::NovaOnly | Self::NovaCpuSnn | Self::NovaGpuSnn | Self::Full)
    }
    
    pub fn uses_gpu_snn(&self) -> bool {
        matches!(self, Self::AnmGpuSnn | Self::NovaGpuSnn | Self::Full)
    }
    
    pub fn uses_tda(&self) -> bool {
        matches!(self, Self::Full)
    }
    
    pub fn uses_rls(&self) -> bool {
        matches!(self, Self::Full)
    }
    
    /// All variants for iteration
    pub fn all() -> &'static [AblationVariant] {
        &[
            Self::AnmOnly,
            Self::AnmGpuSnn,
            Self::NovaOnly,
            Self::NovaCpuSnn,
            Self::NovaGpuSnn,
            Self::Full,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResult {
    pub variant: AblationVariant,
    pub roc_auc: f32,
    pub pr_auc: f32,
    pub success_rate: f32,
    pub top1_accuracy: f32,
    pub time_per_structure: f32,
    pub n_structures: usize,
}

impl AblationResult {
    /// Compute delta from baseline
    pub fn delta_from(&self, baseline: &AblationResult) -> AblationDelta {
        AblationDelta {
            variant: self.variant,
            delta_roc_auc: self.roc_auc - baseline.roc_auc,
            delta_pr_auc: self.pr_auc - baseline.pr_auc,
            delta_success_rate: self.success_rate - baseline.success_rate,
            speedup: baseline.time_per_structure / self.time_per_structure.max(0.001),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AblationDelta {
    pub variant: AblationVariant,
    pub delta_roc_auc: f32,
    pub delta_pr_auc: f32,
    pub delta_success_rate: f32,
    pub speedup: f32,
}

#[derive(Debug, Clone, Default)]
pub struct AblationStudy {
    pub results: Vec<AblationResult>,
}

impl AblationStudy {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add_result(&mut self, result: AblationResult) {
        self.results.push(result);
    }
    
    pub fn get_baseline(&self) -> Option<&AblationResult> {
        self.results.iter().find(|r| r.variant == AblationVariant::AnmOnly)
    }
    
    pub fn get_full(&self) -> Option<&AblationResult> {
        self.results.iter().find(|r| r.variant == AblationVariant::Full)
    }
    
    /// Generate comparison table (markdown)
    pub fn to_markdown_table(&self) -> String {
        let baseline = self.get_baseline();
        
        let mut s = String::new();
        s.push_str("| Variant | ROC AUC | Δ AUC | PR AUC | Success | Time |\n");
        s.push_str("|---------|---------|-------|--------|---------|------|\n");
        
        for r in &self.results {
            let delta = baseline.map(|b| r.roc_auc - b.roc_auc).unwrap_or(0.0);
            let delta_str = if delta > 0.0 { format!("+{:.3}", delta) } 
                           else { format!("{:.3}", delta) };
            
            s.push_str(&format!("| {} | {:.3} | {} | {:.3} | {:.1}% | {:.2}s |\n",
                                r.variant.name(),
                                r.roc_auc,
                                delta_str,
                                r.pr_auc,
                                r.success_rate * 100.0,
                                r.time_per_structure));
        }
        s
    }
    
    /// Generate LaTeX table
    pub fn to_latex_table(&self) -> String {
        let baseline = self.get_baseline();
        
        let mut s = String::new();
        s.push_str("\\begin{table}[h]\n\\centering\n");
        s.push_str("\\caption{Ablation Study Results}\n");
        s.push_str("\\begin{tabular}{lccccc}\n\\toprule\n");
        s.push_str("Variant & ROC AUC & $\\Delta$ & PR AUC & Success & Time \\\\\n");
        s.push_str("\\midrule\n");
        
        for r in &self.results {
            let delta = baseline.map(|b| r.roc_auc - b.roc_auc).unwrap_or(0.0);
            let delta_str = if delta > 0.0 { format!("+{:.3}", delta) }
                           else if delta < 0.0 { format!("{:.3}", delta) }
                           else { "---".to_string() };
            
            s.push_str(&format!("{} & {:.3} & {} & {:.3} & {:.1}\\% & {:.2}s \\\\\n",
                                r.variant.name().replace("_", "\\_"),
                                r.roc_auc,
                                delta_str,
                                r.pr_auc,
                                r.success_rate * 100.0,
                                r.time_per_structure));
        }
        
        s.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        s
    }
}
```

---

### Task 5.3: Failure Case Analysis

**File**: `crates/prism-validation/src/failure_analysis.rs`

```rust
//! Failure case analysis for limitations section
//!
//! Categorizes why certain structures fail to help identify
//! systematic limitations and future work directions.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureReason {
    /// Pocket buried too deep for sampling to reach
    PocketTooDeep,
    
    /// Requires >5Å backbone motion (beyond NOVA capability)
    LargeConformationalChange,
    
    /// Allosteric site distal from active site
    AllostericSite,
    
    /// Pocket is actually crystal packing artifact
    CrystalContact,
    
    /// Multiple annotated pockets causing ambiguity
    MultiplePockets,
    
    /// Structure quality issues (missing atoms, bad geometry)
    PoorStructureQuality,
    
    /// Very small pocket (<100 Å³)
    SmallPocket,
    
    /// Pocket only forms with cofactor/ion
    CofactorDependent,
    
    /// Unknown/unclassified
    Unknown,
}

impl FailureReason {
    pub fn description(&self) -> &'static str {
        match self {
            Self::PocketTooDeep => "Pocket buried beyond sampling reach",
            Self::LargeConformationalChange => "Requires large backbone motion (>5Å)",
            Self::AllostericSite => "Allosteric site distal from active region",
            Self::CrystalContact => "Pocket is crystal packing artifact",
            Self::MultiplePockets => "Multiple pockets cause annotation ambiguity",
            Self::PoorStructureQuality => "Structure has quality issues",
            Self::SmallPocket => "Very small pocket (<100 Å³)",
            Self::CofactorDependent => "Pocket requires cofactor to form",
            Self::Unknown => "Unclassified failure",
        }
    }
    
    pub fn is_fundamental_limitation(&self) -> bool {
        matches!(self, 
            Self::LargeConformationalChange | 
            Self::AllostericSite |
            Self::CofactorDependent)
    }
    
    pub fn is_data_issue(&self) -> bool {
        matches!(self,
            Self::CrystalContact |
            Self::MultiplePockets |
            Self::PoorStructureQuality)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCaseAnalysis {
    pub structure_id: String,
    pub predicted_auc: f32,
    pub reason: FailureReason,
    pub notes: String,
}

#[derive(Debug, Clone, Default)]
pub struct FailureReport {
    pub cases: Vec<FailureCaseAnalysis>,
}

impl FailureReport {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add(&mut self, case: FailureCaseAnalysis) {
        self.cases.push(case);
    }
    
    /// Count by reason
    pub fn count_by_reason(&self) -> Vec<(FailureReason, usize)> {
        let mut counts = std::collections::HashMap::new();
        for case in &self.cases {
            *counts.entry(case.reason).or_insert(0) += 1;
        }
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
    
    /// Generate markdown summary
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("## Failure Case Analysis\n\n");
        s.push_str(&format!("Total failures: {}\n\n", self.cases.len()));
        
        s.push_str("### By Category\n\n");
        for (reason, count) in self.count_by_reason() {
            s.push_str(&format!("- **{}**: {} ({:.0}%)\n", 
                                reason.description(), 
                                count,
                                count as f32 / self.cases.len() as f32 * 100.0));
        }
        
        let fundamental = self.cases.iter()
            .filter(|c| c.reason.is_fundamental_limitation())
            .count();
        let data_issues = self.cases.iter()
            .filter(|c| c.reason.is_data_issue())
            .count();
        
        s.push_str(&format!("\n**Fundamental limitations**: {} ({:.0}%)\n",
                            fundamental, fundamental as f32 / self.cases.len() as f32 * 100.0));
        s.push_str(&format!("**Data issues**: {} ({:.0}%)\n",
                            data_issues, data_issues as f32 / self.cases.len() as f32 * 100.0));
        
        s
    }
}
```

---

### Week 5-6 Checklist

```
□ cryptobench_dataset.rs compiles
□ Dataset loads all 1107 structures
□ Train/test split verified (885/222, no leakage)
□ ablation.rs compiles
□ All 6 ablation variants run
□ Full pipeline > ANM-only by >0.20 AUC
□ failure_analysis.rs compiles
□ Bottom 10% failures categorized
□ Metrics computed on test set only
□ Results serialized to JSON
```

---

## 7. Weeks 7-8: Publication & Final Validation

### Objective

- Generate publication-ready figures and tables
- Final benchmark sweep
- Package for release

### Files to Create

1. `crates/prism-validation/src/publication_outputs.rs`
2. `scripts/generate_figures.py`
3. `results/figures/` directory

---

### Task 7.1: Publication Outputs

**File**: `crates/prism-validation/src/publication_outputs.rs`

```rust
//! Publication-ready output generation
//!
//! Generates:
//! - LaTeX tables for main results
//! - Figure data (JSON/CSV for Python plotting)
//! - Methods section draft

use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub struct PublicationResults {
    pub main_metrics: MainMetrics,
    pub ablation_study: Vec<AblationRow>,
    pub apo_holo_results: Vec<ApoHoloRow>,
    pub failure_summary: FailureSummary,
    pub timing_stats: TimingStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct MainMetrics {
    pub roc_auc: f32,
    pub roc_auc_ci: (f32, f32),
    pub pr_auc: f32,
    pub pr_auc_ci: (f32, f32),
    pub success_rate: f32,
    pub top1_accuracy: f32,
    pub n_test_structures: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AblationRow {
    pub variant: String,
    pub roc_auc: f32,
    pub delta_auc: f32,
    pub pr_auc: f32,
    pub success_rate: f32,
    pub time_seconds: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApoHoloRow {
    pub name: String,
    pub apo_pdb: String,
    pub holo_pdb: String,
    pub start_rmsd: f32,
    pub best_rmsd: f32,
    pub improvement: f32,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FailureSummary {
    pub total_failures: usize,
    pub by_category: Vec<(String, usize)>,
    pub fundamental_limitations: usize,
    pub data_issues: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimingStats {
    pub mean_time_per_structure: f32,
    pub median_time_per_structure: f32,
    pub total_test_time_hours: f32,
    pub gpu_model: String,
    pub peak_vram_gb: f32,
}

impl PublicationResults {
    /// Save all outputs to directory
    pub fn save_all(&self, output_dir: &str) -> Result<()> {
        fs::create_dir_all(output_dir)?;
        
        // JSON for programmatic access
        let json = serde_json::to_string_pretty(self)?;
        fs::write(format!("{}/results.json", output_dir), json)?;
        
        // LaTeX tables
        fs::write(
            format!("{}/table_main_results.tex", output_dir),
            self.generate_main_table()
        )?;
        
        fs::write(
            format!("{}/table_ablation.tex", output_dir),
            self.generate_ablation_table()
        )?;
        
        fs::write(
            format!("{}/table_apo_holo.tex", output_dir),
            self.generate_apo_holo_table()
        )?;
        
        // CSV for plotting
        fs::write(
            format!("{}/ablation_data.csv", output_dir),
            self.generate_ablation_csv()
        )?;
        
        // Methods section
        fs::write(
            format!("{}/methods_section.md", output_dir),
            self.generate_methods()
        )?;
        
        log::info!("Publication outputs saved to {}", output_dir);
        Ok(())
    }
    
    fn generate_main_table(&self) -> String {
        format!(r#"\begin{{table}}[h]
\centering
\caption{{PRISM-ZrO Cryptic Site Detection Performance on CryptoBench}}
\label{{tab:main_results}}
\begin{{tabular}}{{lc}}
\toprule
Metric & Value \\
\midrule
ROC AUC & {:.3} ({:.3}--{:.3}) \\
PR AUC & {:.3} ({:.3}--{:.3}) \\
Success Rate ($\geq$30\% overlap) & {:.1}\% \\
Top-1 Accuracy & {:.1}\% \\
Test Set Size & {} structures \\
\bottomrule
\end{{tabular}}
\end{{table}}
"#,
            self.main_metrics.roc_auc,
            self.main_metrics.roc_auc_ci.0,
            self.main_metrics.roc_auc_ci.1,
            self.main_metrics.pr_auc,
            self.main_metrics.pr_auc_ci.0,
            self.main_metrics.pr_auc_ci.1,
            self.main_metrics.success_rate * 100.0,
            self.main_metrics.top1_accuracy * 100.0,
            self.main_metrics.n_test_structures)
    }
    
    fn generate_ablation_table(&self) -> String {
        let mut s = String::from(r#"\begin{table}[h]
\centering
\caption{Ablation Study: Component Contributions}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Configuration & ROC AUC & $\Delta$ & PR AUC & Success & Time \\
\midrule
"#);
        
        for row in &self.ablation_study {
            let delta = if row.delta_auc > 0.0 {
                format!("+{:.3}", row.delta_auc)
            } else if row.delta_auc < 0.0 {
                format!("{:.3}", row.delta_auc)
            } else {
                "---".to_string()
            };
            
            s.push_str(&format!(
                "{} & {:.3} & {} & {:.3} & {:.1}\\% & {:.2}s \\\\\n",
                row.variant.replace("_", "\\_"),
                row.roc_auc,
                delta,
                row.pr_auc,
                row.success_rate * 100.0,
                row.time_seconds
            ));
        }
        
        s.push_str(r#"\bottomrule
\end{tabular}
\end{table}
"#);
        s
    }
    
    fn generate_apo_holo_table(&self) -> String {
        // Similar LaTeX table generation for apo-holo results
        let mut s = String::from(r#"\begin{table}[h]
\centering
\caption{Apo-Holo Conformational Change Prediction}
\label{tab:apo_holo}
\begin{tabular}{lcccc}
\toprule
Protein & Start & Best & $\Delta$ & Success \\
\midrule
"#);
        
        for row in &self.apo_holo_results {
            s.push_str(&format!(
                "{} & {:.2}\\AA & {:.2}\\AA & {:.2}\\AA & {} \\\\\n",
                row.name.replace("_", "\\_"),
                row.start_rmsd,
                row.best_rmsd,
                row.improvement,
                if row.success { "\\checkmark" } else { "$\\times$" }
            ));
        }
        
        s.push_str(r#"\bottomrule
\end{tabular}
\end{table}
"#);
        s
    }
    
    fn generate_ablation_csv(&self) -> String {
        let mut s = String::from("variant,roc_auc,delta_auc,pr_auc,success_rate,time_seconds\n");
        for row in &self.ablation_study {
            s.push_str(&format!("{},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                                row.variant, row.roc_auc, row.delta_auc,
                                row.pr_auc, row.success_rate, row.time_seconds));
        }
        s
    }
    
    fn generate_methods(&self) -> String {
        format!(r#"# Methods

## Cryptic Site Detection Pipeline

PRISM-ZrO combines neuromorphic computing with enhanced conformational sampling
for cryptic binding site detection.

### Conformational Sampling

Structures were sampled using PRISM-NOVA, a neural Hamiltonian Monte Carlo
implementation running on GPU. Parameters:
- Temperature: 310 K
- Timestep: 2 fs
- Leapfrog steps: 5
- Samples per structure: 500
- Decorrelation steps: 100

### Feature Extraction

Per-residue features (16 dimensions) were computed including:
- Dynamics: burial change, RMSF, variance, neighbor flexibility
- Structural: secondary structure flexibility, B-factor
- Chemical: charge, hydrophobicity, H-bond potential
- Spatial: contact density, SASA change, interface proximity

### Neural Scoring

Features were processed through a 512-neuron GPU-accelerated dendritic
spiking neural network reservoir with RLS online learning (λ=0.99).

### Evaluation

Performance was evaluated on the CryptoBench benchmark:
- {} test structures (held out from training)
- Primary metrics: ROC AUC, PR AUC
- Secondary metrics: Success rate (≥30% overlap), Top-1 accuracy

### Hardware

All experiments were performed on:
- GPU: {}
- Peak VRAM: {:.1} GB
- Mean time per structure: {:.2} seconds
"#,
            self.main_metrics.n_test_structures,
            self.timing_stats.gpu_model,
            self.timing_stats.peak_vram_gb,
            self.timing_stats.mean_time_per_structure)
    }
}
```

---

### Week 7-8 Checklist

```
□ publication_outputs.rs compiles
□ All LaTeX tables generate correctly
□ CSV data exports for plotting
□ ROC curve data exported
□ PR curve data exported
□ Figure generation scripts work
□ Methods section draft complete
□ Final benchmark sweep matches targets
□ All results reproducible from clean state
□ Code packaged for release
□ README updated with usage instructions
```

---

## 8. Complete File Manifest

### New Files (14 total)

| File | Purpose | Week |
|------|---------|------|
| `cryptic_features.rs` | Feature vector definition | 1 |
| `gpu_zro_cryptic_scorer.rs` | GPU 512-neuron scorer | 1 |
| `ensemble_cryptic_model.rs` | Ensemble weight learning | 1 |
| `ensemble_quality_metrics.rs` | Sampling validation | 1 |
| `tests/gpu_scorer_tests.rs` | GPU scorer tests | 2 |
| `pdb_sanitizer.rs` | PDB preprocessing | 3 |
| `nova_cryptic_sampler.rs` | NOVA HMC wrapper | 3 |
| `apo_holo_benchmark.rs` | Conformational validation | 4 |
| `cryptobench_dataset.rs` | Dataset loader | 5 |
| `cryptobench_benchmark.rs` | Full benchmark runner | 5 |
| `ablation.rs` | Ablation study framework | 5 |
| `failure_analysis.rs` | Failure categorization | 6 |
| `publication_outputs.rs` | LaTeX/figure generation | 7 |
| `scripts/generate_figures.py` | Plotting scripts | 8 |

### Files to Modify (3)

| File | Changes |
|------|---------|
| `blind_validation_pipeline.rs` | Add GPU scorer, NOVA options |
| `lib.rs` | Export Phase 6 modules |
| `Cargo.toml` | Add `cuda` feature, dependencies |

---

## 9. Verification Commands

```bash
# Week 0: Setup verification
cargo check -p prism-validation --features cuda

# Week 2: GPU scorer tests
cargo test --release -p prism-validation --features cuda gpu_scorer -- --nocapture
CUDA_VISIBLE_DEVICES="" cargo test test_no_cpu_fallback  # MUST FAIL

# Week 4: NOVA sampling test  
cargo run --release -p prism-validation --bin test-nova-sampler -- \
    --pdb data/test/1ake.pdb --samples 100 --verbose

# Week 4: Apo-holo benchmark
cargo run --release -p prism-validation --bin apo-holo-benchmark -- \
    --data-dir data/benchmarks/apo_holo --output results/apo_holo.json

# Week 6: Full CryptoBench benchmark
cargo run --release -p prism-validation --bin cryptobench-benchmark -- \
    --manifest data/benchmarks/cryptobench/manifest.json \
    --output results/cryptobench_full.json

# Week 6: Ablation study
cargo run --release -p prism-validation --bin ablation-study -- \
    --manifest data/benchmarks/cryptobench/manifest.json \
    --output results/ablation.json

# Week 8: Generate publication outputs
cargo run --release -p prism-validation --bin generate-publication -- \
    --results results/cryptobench_full.json \
    --ablation results/ablation.json \
    --apo-holo results/apo_holo.json \
    --output results/publication/
```

---

## 10. Risk Mitigation

| Risk | Prevention | Recovery |
|------|------------|----------|
| GPU memory overflow | Batch processing | Reduce reservoir to 256 neurons |
| NOVA divergence | Conservative dt (2fs) | Lower temperature, shorter leapfrog |
| RLS instability | λ=0.99, gradient clamp | Soft reset precision matrix |
| Low acceptance | Monitor rate | Increase temperature |
| Dataset issues | Pre-validate all PDBs | Skip corrupt entries |
| Metric regression | Per-structure logging | Bisect to find cause |

---

## Final Success Criteria

```
✅ ROC AUC > 0.70 (target: 0.75)
✅ PR AUC > 0.20 (target: 0.25)
✅ Success Rate > 80% (target: 85%)
✅ Top-1 Accuracy > 85% (target: 90%)
✅ Time < 5s/structure (target: 1s)
✅ Ablation shows >+0.20 AUC improvement
✅ Apo-holo shows >60% success rate
✅ Zero CPU fallback (verified by test)
✅ Zero mock implementations
✅ Publication-ready outputs generated
```

---

**Document End**

*This plan is approved for Phase 6 execution. Begin with Week 0 setup tasks.*
