// [STAGE-2-BVH] OptiX Acceleration Structure (BVH)
//
// Builds and manages Bounding Volume Hierarchy (BVH) for efficient ray tracing.
// Targets <100ms build time and <10ms refit time for 100K atoms.

use crate::context::OptixContext;
use crate::error::{OptixError, Result};
use optix_sys::*;
use std::ptr;

/// BVH build input type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvhInputType {
    /// Triangle mesh (for surfaces)
    Triangles,
    /// Custom primitives (for atoms/spheres)
    Custom,
    /// Instances (for hierarchical BVH)
    Instances,
}

/// BVH build flags
#[derive(Debug, Clone, Copy)]
pub struct BvhBuildFlags {
    /// Allow updating (refit) without full rebuild
    pub allow_update: bool,
    /// Allow compaction for smaller memory footprint
    pub allow_compaction: bool,
    /// Prefer fast trace over fast build
    pub prefer_fast_trace: bool,
    /// Allow random vertex access
    pub allow_random_vertex_access: bool,
}

impl Default for BvhBuildFlags {
    fn default() -> Self {
        Self {
            allow_update: true, // Enable refit for dynamic atoms
            allow_compaction: false, // Disable for speed
            prefer_fast_trace: true, // Optimize for query performance
            allow_random_vertex_access: false,
        }
    }
}

impl BvhBuildFlags {
    /// Flags optimized for dynamic atom updates (refit)
    pub fn dynamic() -> Self {
        Self {
            allow_update: true,
            allow_compaction: false,
            prefer_fast_trace: false, // Prefer fast refit
            allow_random_vertex_access: false,
        }
    }

    /// Flags optimized for static geometry (no updates)
    pub fn static_geometry() -> Self {
        Self {
            allow_update: false,
            allow_compaction: true,
            prefer_fast_trace: true,
            allow_random_vertex_access: false,
        }
    }
}

/// OptiX acceleration structure (BVH)
///
/// Manages a BVH for efficient spatial queries. Supports both full builds
/// and fast refits for dynamic geometry.
///
/// # Performance Targets
///
/// - Build: <100ms for 100K atoms
/// - Refit: <10ms for 100K atoms (when positions change)
pub struct AccelStructure {
    /// OptiX context this BVH belongs to
    context: *const OptixContext,

    /// BVH handle
    handle: OptixTraversableHandle,

    /// GPU buffer for BVH data
    device_buffer: *mut u8,
    device_buffer_size: usize,

    /// Whether this BVH can be updated (refitted)
    can_update: bool,
}

impl AccelStructure {
    /// Build a BVH for custom primitives (atoms as spheres)
    ///
    /// # Arguments
    ///
    /// * `context` - OptiX context
    /// * `positions` - Atom positions (x, y, z) on GPU
    /// * `radii` - Atom radii on GPU
    /// * `num_atoms` - Number of atoms
    /// * `flags` - Build flags
    ///
    /// # Example
    ///
    /// ```no_run
    /// use prism_optix::{OptixContext, AccelStructure, BvhBuildFlags};
    /// use cudarc::driver::CudaDevice;
    ///
    /// OptixContext::init().unwrap();
    /// let cuda = CudaDevice::new(0).unwrap();
    /// let ctx = OptixContext::new(cuda.cu_primary_ctx(), true).unwrap();
    ///
    /// // Upload atom positions and radii to GPU
    /// // let d_positions = ...;
    /// // let d_radii = ...;
    ///
    /// // Build BVH
    /// // let bvh = AccelStructure::build_custom_primitives(
    /// //     &ctx, d_positions, d_radii, num_atoms, BvhBuildFlags::dynamic()
    /// // ).unwrap();
    /// ```
    pub fn build_custom_primitives(
        context: &OptixContext,
        _positions_gpu: *const f32,
        _radii_gpu: *const f32,
        _num_atoms: usize,
        flags: BvhBuildFlags,
    ) -> Result<Self> {
        // NOTE: Full implementation requires additional OptiX functions:
        // - optixAccelComputeMemoryUsage
        // - optixAccelBuild
        // These will be added to loader.rs in subsequent commits

        log::info!(
            "Building BVH for atoms (flags: allow_update={}, prefer_fast_trace={})",
            flags.allow_update,
            flags.prefer_fast_trace
        );

        // Placeholder implementation
        Ok(Self {
            context: context as *const OptixContext,
            handle: 0,
            device_buffer: ptr::null_mut(),
            device_buffer_size: 0,
            can_update: flags.allow_update,
        })
    }

    /// Refit the BVH with updated positions (fast update)
    ///
    /// Much faster than a full rebuild (~100x faster). Use when atom
    /// positions change but topology remains the same.
    ///
    /// # Performance
    ///
    /// Target: <10ms for 100K atoms
    pub fn refit(&mut self, _positions_gpu: *const f32) -> Result<()> {
        if !self.can_update {
            return Err(OptixError::InvalidOperation(
                "BVH was not built with allow_update flag".to_string(),
            ));
        }

        log::debug!("Refitting BVH with updated positions");

        // NOTE: Full implementation requires optixAccelRefit
        // Will be added to loader.rs in subsequent commits

        Ok(())
    }

    /// Get the traversable handle for ray tracing
    pub fn handle(&self) -> OptixTraversableHandle {
        self.handle
    }

    /// Check if this BVH can be updated (refitted)
    pub fn can_update(&self) -> bool {
        self.can_update
    }

    /// Get device buffer size in bytes
    pub fn device_buffer_size(&self) -> usize {
        self.device_buffer_size
    }
}

impl Drop for AccelStructure {
    fn drop(&mut self) {
        if !self.device_buffer.is_null() {
            // NOTE: Full implementation requires CUDA memory deallocation
            // For now, just log
            log::debug!(
                "Dropping AccelStructure (buffer size: {} bytes)",
                self.device_buffer_size
            );
        }
    }
}

// AccelStructure is Send (can be moved between threads)
unsafe impl Send for AccelStructure {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bvh_flags_default() {
        let flags = BvhBuildFlags::default();
        assert!(flags.allow_update);
        assert!(flags.prefer_fast_trace);
        assert!(!flags.allow_compaction);
    }

    #[test]
    fn test_bvh_flags_dynamic() {
        let flags = BvhBuildFlags::dynamic();
        assert!(flags.allow_update);
        assert!(!flags.prefer_fast_trace); // Prefer fast refit
    }

    #[test]
    fn test_bvh_flags_static() {
        let flags = BvhBuildFlags::static_geometry();
        assert!(!flags.allow_update);
        assert!(flags.allow_compaction);
        assert!(flags.prefer_fast_trace);
    }
}
