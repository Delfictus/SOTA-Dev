// [STAGE-2-RT-PROBE] RT Probe Engine
//
// RT probe engine for RTX 5080's 84 RT cores.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use prism_optix::{AccelStructure, BvhBuildFlags, OptixContext};

/// RT probe configuration
#[derive(Debug, Clone)]
pub struct RtProbeConfig {
    pub probe_interval: i32,
    pub rays_per_point: usize,
    pub attention_points: usize,
    pub bvh_refit_threshold: f32,
    pub track_solvation: bool,
    pub track_aromatic_lif: bool,
}

impl Default for RtProbeConfig {
    fn default() -> Self {
        Self {
            probe_interval: 100,
            rays_per_point: 256,
            attention_points: 50,
            bvh_refit_threshold: 0.5,
            track_solvation: false,
            track_aromatic_lif: true,
        }
    }
}

/// RT probe snapshot
#[derive(Debug, Clone)]
pub struct RtProbeSnapshot {
    pub timestep: i32,
    pub hit_distances: Vec<f32>,
    pub void_detected: bool,
    pub solvation_variance: Option<f32>,
    pub aromatic_lif_count: usize,
}

/// RT Probe Engine
pub struct RtProbeEngine {
    optix_ctx: OptixContext,
    bvh_protein: Option<AccelStructure>,
    config: RtProbeConfig,
    max_displacement: f32,
    snapshots: Vec<RtProbeSnapshot>,
}

impl RtProbeEngine {
    pub fn new(optix_ctx: OptixContext, config: RtProbeConfig) -> Result<Self> {
        Ok(Self {
            optix_ctx,
            bvh_protein: None,
            config,
            max_displacement: 0.0,
            snapshots: Vec::new(),
        })
    }

    pub fn build_protein_bvh(
        &mut self,
        positions_gpu: &CudaSlice<f32>,
        radii_gpu: &CudaSlice<f32>,
        num_atoms: usize,
    ) -> Result<()> {
        let bvh = AccelStructure::build_custom_primitives(
            &self.optix_ctx,
            positions_gpu.device_ptr() as *const f32,
            radii_gpu.device_ptr() as *const f32,
            num_atoms,
            BvhBuildFlags::dynamic(),
        )?;
        self.bvh_protein = Some(bvh);
        Ok(())
    }

    pub fn needs_refit(&self, displacement: f32) -> bool {
        displacement > self.config.bvh_refit_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rt_probe_config_default() {
        let config = RtProbeConfig::default();
        assert_eq!(config.probe_interval, 100);
    }
}
