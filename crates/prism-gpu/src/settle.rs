//! SETTLE - Analytical constraint solver for rigid water
//!
//! Maintains TIP3P water geometry exactly during MD integration:
//! - OH bond length: 0.9572 Å
//! - HH distance: 1.5136 Å (from geometry)
//! - HOH angle: 104.52°
//!
//! The SETTLE algorithm analytically solves the constraint equations
//! in a single step, making it more efficient than iterative methods
//! like SHAKE for rigid water.
//!
//! Reference: Miyamoto & Kollman (1992) J. Comput. Chem. 13:952-962

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// TIP3P water geometry parameters
pub const TIP3P_OH_DISTANCE: f32 = 0.9572;  // Å
pub const TIP3P_HH_DISTANCE: f32 = 1.5136;  // Å (computed from angle)
pub const TIP3P_MASS_O: f32 = 15.9994;      // g/mol
pub const TIP3P_MASS_H: f32 = 1.008;        // g/mol

/// SETTLE constraint solver for rigid water molecules
pub struct Settle {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    // Kernel
    settle_kernel: CudaFunction,

    // Device buffers
    d_water_indices: CudaSlice<i32>,  // [n_waters * 3] - O, H1, H2 indices for each water
    d_old_positions: CudaSlice<f32>, // [n_total_atoms * 3] - positions before integration

    // Water molecule parameters
    n_waters: usize,
    n_total_atoms: usize,

    // Masses for SETTLE (precomputed)
    mass_o: f32,
    mass_h: f32,
    inv_total_mass: f32,
    ra: f32,  // Distance from O to center of mass
    rb: f32,  // Distance from H to center of mass along OH
    rc: f32,  // Half HH distance

    // Target distances (squared for efficiency)
    roh2: f32,  // OH² target
    rhh2: f32,  // HH² target
}

impl Settle {
    /// Create a new SETTLE constraint solver
    ///
    /// # Arguments
    /// * `context` - CUDA context
    /// * `water_oxygen_indices` - Indices of oxygen atoms in each water molecule
    ///   The hydrogens are assumed to be at indices O+1 and O+2
    /// * `n_total_atoms` - Total number of atoms in the system
    pub fn new(
        context: Arc<CudaContext>,
        water_oxygen_indices: &[usize],
        n_total_atoms: usize,
    ) -> Result<Self> {
        log::info!(
            "🌊 Initializing SETTLE for {} water molecules",
            water_oxygen_indices.len()
        );

        let stream = context.default_stream();
        let n_waters = water_oxygen_indices.len();

        // Load SETTLE PTX module
        let ptx_path = "crates/prism-gpu/target/ptx/settle.ptx";
        let ptx = Ptx::from_file(ptx_path);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load SETTLE PTX from {}", ptx_path))?;

        // Load kernel
        let settle_kernel = module
            .load_function("settle_constraints")
            .context("Failed to load settle_constraints")?;

        // Build water index array: [O0, H0_1, H0_2, O1, H1_1, H1_2, ...]
        let mut water_indices = Vec::with_capacity(n_waters * 3);
        for &o_idx in water_oxygen_indices {
            water_indices.push(o_idx as i32);
            water_indices.push((o_idx + 1) as i32);  // H1
            water_indices.push((o_idx + 2) as i32);  // H2
        }

        // Allocate device buffers
        let d_water_indices = stream.alloc_zeros::<i32>(water_indices.len().max(1))?;
        let d_old_positions = stream.alloc_zeros::<f32>(n_total_atoms * 3)?;

        // Upload water indices
        let mut d_water_indices = d_water_indices;
        if !water_indices.is_empty() {
            stream.memcpy_htod(&water_indices, &mut d_water_indices)?;
        }

        // Precompute SETTLE geometry parameters
        let mass_o = TIP3P_MASS_O;
        let mass_h = TIP3P_MASS_H;
        let total_mass = mass_o + 2.0 * mass_h;
        let inv_total_mass = 1.0 / total_mass;

        // SETTLE canonical frame geometry:
        // - O is at (0, 0, -ra) - distance ra from COM along negative bisector
        // - H1 is at (-rc, 0, rb) - distance rb along positive bisector, rc sideways
        // - H2 is at (+rc, 0, rb) - symmetric with H1
        //
        // From center of mass constraint: mO*(-ra) + 2*mH*rb = 0
        // Therefore: rb = ra * mO / (2*mH)
        //
        // From OH distance constraint: OH² = (ra+rb)² + rc²
        // Solving: ra + rb = sqrt(OH² - rc²) = OH*cos(half_angle)
        //
        // Combined: ra*(1 + mO/(2*mH)) = OH*cos(half_angle)
        //           ra = OH*cos(half_angle) / (1 + mO/(2*mH))
        //           ra = OH*cos(half_angle) * 2*mH / (2*mH + mO)
        //           ra = OH*cos(half_angle) * 2*mH / total_mass

        let half_angle = 104.52_f32.to_radians() / 2.0;
        let ra = TIP3P_OH_DISTANCE * half_angle.cos() * 2.0 * mass_h * inv_total_mass;

        // rb from COM constraint: mO*ra = 2*mH*rb
        let rb = ra * mass_o / (2.0 * mass_h);

        // rc = half of HH distance
        let rc = TIP3P_HH_DISTANCE / 2.0;

        // Target distances squared
        let roh2 = TIP3P_OH_DISTANCE * TIP3P_OH_DISTANCE;
        let rhh2 = TIP3P_HH_DISTANCE * TIP3P_HH_DISTANCE;

        log::info!(
            "✅ SETTLE initialized: {} waters, ra={:.4}, rb={:.4}, rc={:.4}",
            n_waters, ra, rb, rc
        );

        Ok(Self {
            context,
            stream,
            module,
            settle_kernel,
            d_water_indices,
            d_old_positions,
            n_waters,
            n_total_atoms,
            mass_o,
            mass_h,
            inv_total_mass,
            ra,
            rb,
            rc,
            roh2,
            rhh2,
        })
    }

    /// Store old positions before integration step
    ///
    /// Call this BEFORE the velocity Verlet integration to save
    /// positions for the constraint projection.
    pub fn save_positions(&mut self, d_positions: &CudaSlice<f32>) -> Result<()> {
        // Copy current positions to old positions buffer
        // This is needed because SETTLE needs the old positions to compute
        // the constraint projection direction
        self.stream.memcpy_dtod(d_positions, &mut self.d_old_positions)?;
        Ok(())
    }

    /// Apply SETTLE constraints to positions
    ///
    /// This projects the new positions back onto the constraint surface
    /// while conserving momentum.
    ///
    /// # Arguments
    /// * `d_positions` - Device buffer with positions AFTER integration
    ///   Will be modified in-place to satisfy constraints
    /// * `dt` - Timestep (for velocity correction if needed)
    pub fn apply(&mut self, d_positions: &mut CudaSlice<f32>, _dt: f32) -> Result<()> {
        if self.n_waters == 0 {
            return Ok(());
        }

        let threads = 256;
        let blocks = (self.n_waters + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_waters_i32 = self.n_waters as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.settle_kernel);
            builder.arg(d_positions);
            builder.arg(&self.d_old_positions);
            builder.arg(&self.d_water_indices);
            builder.arg(&n_waters_i32);
            builder.arg(&self.mass_o);
            builder.arg(&self.mass_h);
            builder.arg(&self.ra);
            builder.arg(&self.rb);
            builder.arg(&self.rc);
            builder.arg(&self.roh2);
            builder.arg(&self.rhh2);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    /// Get number of water molecules
    pub fn n_waters(&self) -> usize {
        self.n_waters
    }

    /// Check constraint violations (for debugging)
    ///
    /// Returns (max_oh_violation, max_hh_violation) in Angstroms
    pub fn check_constraints(&mut self, d_positions: &CudaSlice<f32>) -> Result<(f32, f32)> {
        // Download positions
        let mut positions = vec![0.0f32; self.n_total_atoms * 3];
        self.stream.memcpy_dtoh(d_positions, &mut positions)?;

        // Download water indices
        let mut water_indices = vec![0i32; self.n_waters * 3];
        self.stream.memcpy_dtoh(&self.d_water_indices, &mut water_indices)?;

        let mut max_oh_violation = 0.0f32;
        let mut max_hh_violation = 0.0f32;

        for w in 0..self.n_waters {
            let o = water_indices[w * 3] as usize;
            let h1 = water_indices[w * 3 + 1] as usize;
            let h2 = water_indices[w * 3 + 2] as usize;

            // Get positions
            let ox = positions[o * 3];
            let oy = positions[o * 3 + 1];
            let oz = positions[o * 3 + 2];

            let h1x = positions[h1 * 3];
            let h1y = positions[h1 * 3 + 1];
            let h1z = positions[h1 * 3 + 2];

            let h2x = positions[h2 * 3];
            let h2y = positions[h2 * 3 + 1];
            let h2z = positions[h2 * 3 + 2];

            // Compute distances
            let dx1 = h1x - ox;
            let dy1 = h1y - oy;
            let dz1 = h1z - oz;
            let oh1 = (dx1 * dx1 + dy1 * dy1 + dz1 * dz1).sqrt();

            let dx2 = h2x - ox;
            let dy2 = h2y - oy;
            let dz2 = h2z - oz;
            let oh2 = (dx2 * dx2 + dy2 * dy2 + dz2 * dz2).sqrt();

            let dx12 = h2x - h1x;
            let dy12 = h2y - h1y;
            let dz12 = h2z - h1z;
            let hh = (dx12 * dx12 + dy12 * dy12 + dz12 * dz12).sqrt();

            // Check violations
            let oh1_violation = (oh1 - TIP3P_OH_DISTANCE).abs();
            let oh2_violation = (oh2 - TIP3P_OH_DISTANCE).abs();
            let hh_violation = (hh - TIP3P_HH_DISTANCE).abs();

            max_oh_violation = max_oh_violation.max(oh1_violation).max(oh2_violation);
            max_hh_violation = max_hh_violation.max(hh_violation);
        }

        Ok((max_oh_violation, max_hh_violation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tip3p_geometry() {
        // HH distance should be consistent with OH and angle
        let expected_hh = 2.0 * TIP3P_OH_DISTANCE * (104.52_f32.to_radians() / 2.0).sin();
        assert!((TIP3P_HH_DISTANCE - expected_hh).abs() < 0.001);
    }

    #[test]
    fn test_settle_mass_parameters() {
        let total_mass = TIP3P_MASS_O + 2.0 * TIP3P_MASS_H;
        let inv_total_mass = 1.0 / total_mass;

        // Center of mass should be closer to O than to H
        let half_angle = 104.52_f32.to_radians() / 2.0;
        let ra = TIP3P_OH_DISTANCE * half_angle.cos() * 2.0 * TIP3P_MASS_H * inv_total_mass;

        // ra should be small positive (O is near COM)
        assert!(ra > 0.0 && ra < TIP3P_OH_DISTANCE * 0.2);
    }
}
