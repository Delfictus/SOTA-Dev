// [STAGE-2-RT-PROBE] RT Probe Engine
//
// RT probe engine for RTX 5080's 84 RT cores.
// Uses OptiX built-in spheres for optimal molecular ray tracing.

use anyhow::Result;
use cudarc::driver::sys::CUdeviceptr;
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
    #[allow(dead_code)]
    max_displacement: f32,
    #[allow(dead_code)]
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

    /// Build BVH for protein atoms using OptiX built-in spheres
    ///
    /// # Arguments
    ///
    /// * `positions_gpu` - Device pointer to atom positions (float3: x, y, z per atom)
    /// * `radii_gpu` - Device pointer to atom radii (float per atom)
    /// * `num_atoms` - Number of atoms
    ///
    /// # Performance
    ///
    /// Target: <100ms for 100K atoms using OptiX hardware BVH build
    ///
    /// # Safety
    ///
    /// Caller must ensure device pointers are valid and point to GPU memory with:
    /// - positions_gpu: `num_atoms * 3` floats (x, y, z per atom)
    /// - radii_gpu: `num_atoms` floats
    pub fn build_protein_bvh(
        &mut self,
        positions_gpu: CUdeviceptr,
        radii_gpu: CUdeviceptr,
        num_atoms: usize,
    ) -> Result<()> {
        log::info!(
            "Building protein BVH: {} atoms using OptiX built-in spheres",
            num_atoms
        );

        // Build BVH using OptiX spheres (dynamic flags for refit support)
        let bvh = AccelStructure::build_spheres(
            &self.optix_ctx,
            positions_gpu,
            radii_gpu,
            num_atoms,
            BvhBuildFlags::dynamic(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to build protein BVH: {}", e))?;

        log::info!(
            "✅ Protein BVH built: {} spheres, {} bytes",
            bvh.num_spheres(),
            bvh.device_buffer_size()
        );

        self.bvh_protein = Some(bvh);
        Ok(())
    }

    /// Refit BVH with updated atom positions (fast update)
    ///
    /// Much faster than full rebuild (~10-100x). Use when positions change
    /// but atom count remains the same.
    ///
    /// # Performance
    ///
    /// Target: <10ms for 100K atoms
    ///
    /// # Safety
    ///
    /// Device pointers must point to same number of atoms as original build.
    pub fn refit_bvh(
        &mut self,
        positions_gpu: CUdeviceptr,
        radii_gpu: CUdeviceptr,
    ) -> Result<()> {
        let bvh = self
            .bvh_protein
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No BVH to refit - call build_protein_bvh first"))?;

        bvh.refit(positions_gpu, radii_gpu)
            .map_err(|e| anyhow::anyhow!("BVH refit failed: {}", e))?;

        log::debug!("BVH refitted successfully");
        Ok(())
    }

    /// Check if BVH needs refit based on displacement threshold
    pub fn needs_refit(&self, displacement: f32) -> bool {
        displacement > self.config.bvh_refit_threshold
    }

    /// Get the BVH traversable handle for ray tracing
    pub fn bvh_handle(&self) -> Option<u64> {
        self.bvh_protein.as_ref().map(|bvh| bvh.handle())
    }

    /// Check if BVH is built and ready for ray tracing
    pub fn has_bvh(&self) -> bool {
        self.bvh_protein.is_some()
    }

    /// Get current configuration
    pub fn config(&self) -> &RtProbeConfig {
        &self.config
    }

    /// Get collected snapshots
    pub fn snapshots(&self) -> &[RtProbeSnapshot] {
        &self.snapshots
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
