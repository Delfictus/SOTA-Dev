//! Composite druggability scoring
//!
//! GPU-accelerated batch scoring with optional FluxNet RL weight learning.

use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use prism_gpu::{context::GpuContext, global_context::GlobalGpuContext, LbsGpu};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub volume: f64,
    pub hydrophobicity: f64,
    pub enclosure: f64,
    pub depth: f64,
    pub hbond_capacity: f64,
    pub flexibility: f64,
    pub conservation: f64,
    pub topology: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            volume: 0.15,
            hydrophobicity: 0.20,
            enclosure: 0.15,
            depth: 0.15,
            hbond_capacity: 0.10,
            flexibility: 0.05,
            conservation: 0.10,
            topology: 0.10,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DrugabilityClass {
    HighlyDruggable,
    Druggable,
    DifficultTarget,
    Undruggable,
}

impl Default for DrugabilityClass {
    fn default() -> Self {
        DrugabilityClass::Undruggable
    }
}

impl DrugabilityClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            DrugabilityClass::HighlyDruggable => "Highly Druggable",
            DrugabilityClass::Druggable => "Druggable",
            DrugabilityClass::DifficultTarget => "Difficult Target",
            DrugabilityClass::Undruggable => "Undruggable",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Components {
    pub volume: f64,
    pub hydro: f64,
    pub enclosure: f64,
    pub depth: f64,
    pub hbond: f64,
    pub flex: f64,
    pub cons: f64,
    pub topo: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DruggabilityScore {
    pub total: f64,
    pub classification: DrugabilityClass,
    pub components: Components,
}

pub struct DruggabilityScorer {
    weights: ScoringWeights,
}

impl DruggabilityScorer {
    pub fn new(weights: ScoringWeights) -> Self {
        Self { weights }
    }

    pub fn score(&self, pocket: &crate::pocket::Pocket) -> DruggabilityScore {
        let volume = self.score_volume(pocket.volume);
        let hydro = self.score_hydrophobicity(pocket.mean_hydrophobicity);
        let enclosure = self.score_enclosure(pocket.enclosure_ratio);
        let depth = self.score_depth(pocket.mean_depth);
        let hbond = self.score_hbond(pocket.hbond_donors, pocket.hbond_acceptors);
        let flex = self.score_flexibility(pocket.mean_flexibility);
        let cons = self.score_conservation(pocket.mean_conservation);
        let topo = pocket.persistence_score;

        let total = self.weights.volume * volume
            + self.weights.hydrophobicity * hydro
            + self.weights.enclosure * enclosure
            + self.weights.depth * depth
            + self.weights.hbond_capacity * hbond
            + self.weights.flexibility * flex
            + self.weights.conservation * cons
            + self.weights.topology * topo;

        DruggabilityScore {
            total,
            classification: self.classify(total),
            components: Components {
                volume,
                hydro,
                enclosure,
                depth,
                hbond,
                flex,
                cons,
                topo,
            },
        }
    }

    fn score_volume(&self, v: f64) -> f64 {
        // Sigmoid centered near 650 Å^3, clamped for extremely large pockets
        let x = (v - 650.0) / 250.0;
        1.0 / (1.0 + (-x).exp()).clamp(0.0, 1.0)
    }

    fn score_hydrophobicity(&self, h: f64) -> f64 {
        ((h + 4.5) / 9.0).clamp(0.0, 1.0)
    }

    fn score_enclosure(&self, e: f64) -> f64 {
        // prefer partially enclosed pockets; too enclosed can be penalized slightly
        let clamped = e.clamp(0.0, 1.0);
        if clamped < 0.2 {
            clamped * 0.5
        } else if clamped > 0.9 {
            0.9 - (clamped - 0.9) * 0.8
        } else {
            clamped
        }
    }

    fn score_depth(&self, d: f64) -> f64 {
        let normalized = (d / 12.0).clamp(0.0, 1.2);
        (1.0 / (1.0 + (-4.0 * (normalized - 0.5)).exp())).clamp(0.0, 1.0)
    }

    fn score_hbond(&self, donors: usize, acceptors: usize) -> f64 {
        let total = donors + acceptors;
        (total as f64 / 10.0).min(1.0)
    }

    fn score_flexibility(&self, flex: f64) -> f64 {
        (1.0 - (flex / 100.0)).clamp(0.0, 1.0)
    }

    fn score_conservation(&self, cons: f64) -> f64 {
        cons.clamp(0.0, 1.0)
    }

    fn classify(&self, score: f64) -> DrugabilityClass {
        if score >= 0.7 {
            DrugabilityClass::HighlyDruggable
        } else if score >= 0.5 {
            DrugabilityClass::Druggable
        } else if score >= 0.3 {
            DrugabilityClass::DifficultTarget
        } else {
            DrugabilityClass::Undruggable
        }
    }

    /// GPU-accelerated batch scoring for multiple pockets
    #[cfg(feature = "cuda")]
    pub fn score_batch_gpu(
        &self,
        pockets: &[crate::pocket::Pocket],
        gpu_ctx: &GpuContext,
    ) -> Result<Vec<DruggabilityScore>, crate::LbsError> {
        if pockets.is_empty() {
            return Ok(Vec::new());
        }

        let n = pockets.len();

        // Prepare component vectors
        let volume: Vec<f32> = pockets.iter().map(|p| self.score_volume(p.volume) as f32).collect();
        let hydro: Vec<f32> = pockets.iter().map(|p| self.score_hydrophobicity(p.mean_hydrophobicity) as f32).collect();
        let enclosure: Vec<f32> = pockets.iter().map(|p| self.score_enclosure(p.enclosure_ratio) as f32).collect();
        let depth: Vec<f32> = pockets.iter().map(|p| self.score_depth(p.mean_depth) as f32).collect();
        let hbond: Vec<f32> = pockets.iter().map(|p| self.score_hbond(p.hbond_donors, p.hbond_acceptors) as f32).collect();
        let flex: Vec<f32> = pockets.iter().map(|p| self.score_flexibility(p.mean_flexibility) as f32).collect();
        let cons: Vec<f32> = pockets.iter().map(|p| self.score_conservation(p.mean_conservation) as f32).collect();
        let topo: Vec<f32> = pockets.iter().map(|p| p.persistence_score as f32).collect();

        let weights: [f32; 8] = [
            self.weights.volume as f32,
            self.weights.hydrophobicity as f32,
            self.weights.enclosure as f32,
            self.weights.depth as f32,
            self.weights.hbond_capacity as f32,
            self.weights.flexibility as f32,
            self.weights.conservation as f32,
            self.weights.topology as f32,
        ];

        // Try to use pre-loaded LbsGpu from GlobalGpuContext (zero PTX overhead)
        let gpu_scores = if let Some(lbs_gpu) = GlobalGpuContext::try_get().ok().and_then(|g| g.lbs_locked()) {
            log::debug!("Using pre-loaded LbsGpu for druggability scoring (zero PTX overhead)");
            lbs_gpu.druggability_score(
                &volume, &hydro, &enclosure, &depth, &hbond, &flex, &cons, &topo, weights
            )
        } else {
            log::debug!("GlobalGpuContext LbsGpu not available, creating new instance");
            let lbs_gpu = LbsGpu::new(gpu_ctx.device().clone(), &gpu_ctx.ptx_dir())
                .map_err(|e| crate::LbsError::Gpu(format!("Failed to init LbsGpu: {}", e)))?;
            lbs_gpu.druggability_score(
                &volume, &hydro, &enclosure, &depth, &hbond, &flex, &cons, &topo, weights
            )
        }.map_err(|e| crate::LbsError::Gpu(format!("GPU scoring failed: {}", e)))?;

        // Assemble results with components
        let mut results = Vec::with_capacity(n);
        for (i, &total) in gpu_scores.iter().enumerate() {
            results.push(DruggabilityScore {
                total: total as f64,
                classification: self.classify(total as f64),
                components: Components {
                    volume: volume[i] as f64,
                    hydro: hydro[i] as f64,
                    enclosure: enclosure[i] as f64,
                    depth: depth[i] as f64,
                    hbond: hbond[i] as f64,
                    flex: flex[i] as f64,
                    cons: cons[i] as f64,
                    topo: topo[i] as f64,
                },
            });
        }

        Ok(results)
    }

    /// Get weights as array for RL optimization
    pub fn weights_as_array(&self) -> [f64; 8] {
        [
            self.weights.volume,
            self.weights.hydrophobicity,
            self.weights.enclosure,
            self.weights.depth,
            self.weights.hbond_capacity,
            self.weights.flexibility,
            self.weights.conservation,
            self.weights.topology,
        ]
    }

    /// Update weights from array (for RL optimization)
    pub fn update_weights(&mut self, new_weights: [f64; 8]) {
        self.weights.volume = new_weights[0].clamp(0.0, 1.0);
        self.weights.hydrophobicity = new_weights[1].clamp(0.0, 1.0);
        self.weights.enclosure = new_weights[2].clamp(0.0, 1.0);
        self.weights.depth = new_weights[3].clamp(0.0, 1.0);
        self.weights.hbond_capacity = new_weights[4].clamp(0.0, 1.0);
        self.weights.flexibility = new_weights[5].clamp(0.0, 1.0);
        self.weights.conservation = new_weights[6].clamp(0.0, 1.0);
        self.weights.topology = new_weights[7].clamp(0.0, 1.0);

        // Normalize to sum to 1.0
        let sum: f64 = self.weights.volume + self.weights.hydrophobicity +
            self.weights.enclosure + self.weights.depth + self.weights.hbond_capacity +
            self.weights.flexibility + self.weights.conservation + self.weights.topology;

        if sum > 0.0 {
            self.weights.volume /= sum;
            self.weights.hydrophobicity /= sum;
            self.weights.enclosure /= sum;
            self.weights.depth /= sum;
            self.weights.hbond_capacity /= sum;
            self.weights.flexibility /= sum;
            self.weights.conservation /= sum;
            self.weights.topology /= sum;
        }
    }
}
