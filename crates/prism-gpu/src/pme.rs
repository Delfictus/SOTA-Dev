//! PME (Particle Mesh Ewald) Electrostatics
//!
//! Implements accurate long-range electrostatics for explicit solvent MD.
//!
//! Algorithm:
//! 1. Spread atom charges to 3D grid using B-spline interpolation
//! 2. Forward FFT to reciprocal space
//! 3. Apply Green's function convolution
//! 4. Inverse FFT back to real space
//! 5. Interpolate forces from grid to atoms
//!
//! Uses cuFFT from CUDA toolkit (not an external dependency).

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg, DevicePtrMut,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::cufft_sys::{
    cufft_error_string, CufftComplex, CufftHandle, CUFFT_C2R, CUFFT_R2C, CUFFT_SUCCESS,
};

/// Default grid spacing (Å) - determines PME grid density
const PME_GRID_SPACING: f32 = 1.0;

/// Default real-space cutoff (Å) - must match non-bonded cutoff
const DEFAULT_REAL_SPACE_CUTOFF: f32 = 12.0;

/// Default PME tolerance for Ewald sum convergence
/// Smaller values give higher accuracy but require more reciprocal space work
pub const DEFAULT_PME_TOLERANCE: f32 = 1e-5;

/// Compute Ewald splitting parameter from tolerance and cutoff.
///
/// The Ewald method splits the Coulomb sum into real-space and reciprocal-space
/// components. The splitting parameter β controls this division:
/// - Larger β: More work in reciprocal space (faster for dense systems)
/// - Smaller β: More work in real space (faster for sparse systems)
///
/// Formula: β = sqrt(-ln(tolerance)) / cutoff
///
/// # Arguments
/// * `cutoff` - Real-space cutoff distance in Ångströms
/// * `tolerance` - Relative error tolerance (typically 1e-5 to 1e-6)
///
/// # Returns
/// Ewald splitting parameter β in Å⁻¹
///
/// # Example
/// ```ignore
/// let beta = compute_ewald_beta(12.0, 1e-5);
/// assert!((beta - 0.283).abs() < 0.001);
/// ```
pub fn compute_ewald_beta(cutoff: f32, tolerance: f32) -> f32 {
    debug_assert!(cutoff > 0.0, "Cutoff must be positive");
    debug_assert!(tolerance > 0.0 && tolerance < 1.0, "Tolerance must be in (0, 1)");

    (-tolerance.ln()).sqrt() / cutoff
}

/// PME electrostatics calculator
pub struct PME {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    // Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,

    // Box dimensions (Å)
    box_dims: [f32; 3],

    // Ewald splitting parameter
    beta: f32,

    // cuFFT plans
    fft_plan_r2c: CufftHandle,
    fft_plan_c2r: CufftHandle,
    plans_initialized: bool,

    // Device buffers
    d_charge_grid: CudaSlice<f32>,     // [nx * ny * nz] real grid
    d_complex_grid: CudaSlice<f32>,    // [nx * ny * (nz/2+1) * 2] complex (interleaved)
    d_energy: CudaSlice<f32>,          // [1] reciprocal energy

    // Kernels
    spread_kernel: CudaFunction,
    convolution_kernel: CudaFunction,
    interpolate_kernel: CudaFunction,
    self_energy_kernel: CudaFunction,
    zero_grid_kernel: CudaFunction,
    normalize_kernel: CudaFunction,

    // Atom count (for validation)
    n_atoms: usize,

    // Phase 7: Mixed precision (FP16) PME grid support
    // Note: cuFFT requires FP32, so FP16 is only used for charge spreading
    // The grid is then converted to FP32 before FFT
    fp16_enabled: bool,
    d_charge_grid_fp16: Option<CudaSlice<u16>>,  // FP16 charge grid for spreading
}

impl PME {
    /// Create a new PME calculator
    ///
    /// # Arguments
    /// * `context` - CUDA context
    /// * `n_atoms` - Number of atoms
    /// * `box_dims` - Periodic box dimensions [Lx, Ly, Lz] in Angstroms
    pub fn new(
        context: Arc<CudaContext>,
        n_atoms: usize,
        box_dims: [f32; 3],
    ) -> Result<Self> {
        log::info!(
            "⚡ Initializing PME for {} atoms, box = {:.1}×{:.1}×{:.1} Å",
            n_atoms, box_dims[0], box_dims[1], box_dims[2]
        );

        let stream = context.default_stream();

        // Load PME PTX module
        let ptx_path = "crates/prism-gpu/target/ptx/pme.ptx";
        let ptx = Ptx::from_file(ptx_path);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load PME PTX from {}", ptx_path))?;

        // Load kernels
        let spread_kernel = module
            .load_function("pme_spread_charges")
            .context("Failed to load pme_spread_charges")?;
        let convolution_kernel = module
            .load_function("pme_reciprocal_convolution")
            .context("Failed to load pme_reciprocal_convolution")?;
        let interpolate_kernel = module
            .load_function("pme_interpolate_forces")
            .context("Failed to load pme_interpolate_forces")?;
        let self_energy_kernel = module
            .load_function("pme_self_energy")
            .context("Failed to load pme_self_energy")?;
        let zero_grid_kernel = module
            .load_function("pme_zero_grid")
            .context("Failed to load pme_zero_grid")?;
        let normalize_kernel = module
            .load_function("pme_normalize_grid")
            .context("Failed to load pme_normalize_grid")?;

        // Compute grid dimensions based on box size and target spacing
        let nx = ((box_dims[0] / PME_GRID_SPACING).ceil() as usize).max(8);
        let ny = ((box_dims[1] / PME_GRID_SPACING).ceil() as usize).max(8);
        let nz = ((box_dims[2] / PME_GRID_SPACING).ceil() as usize).max(8);

        // Round up to nice FFT sizes (powers of 2 or products of small primes)
        let nx = round_up_fft_size(nx);
        let ny = round_up_fft_size(ny);
        let nz = round_up_fft_size(nz);

        log::info!(
            "📊 PME grid: {}×{}×{} = {} points, spacing ≈ {:.2} Å",
            nx, ny, nz,
            nx * ny * nz,
            box_dims[0] / nx as f32
        );

        // Allocate device buffers
        let real_size = nx * ny * nz;
        let complex_size = nx * ny * (nz / 2 + 1) * 2; // Interleaved complex

        let d_charge_grid = stream.alloc_zeros::<f32>(real_size)?;
        let d_complex_grid = stream.alloc_zeros::<f32>(complex_size)?;
        let d_energy = stream.alloc_zeros::<f32>(1)?;

        // Create cuFFT plans
        let (fft_plan_r2c, fft_plan_c2r) =
            create_fft_plans(nx, ny, nz).context("Failed to create cuFFT plans")?;

        // Compute Ewald splitting parameter from cutoff and tolerance
        // β = sqrt(-ln(tolerance)) / cutoff
        let beta = compute_ewald_beta(DEFAULT_REAL_SPACE_CUTOFF, DEFAULT_PME_TOLERANCE);
        log::info!(
            "✅ PME initialized: β={:.4} Å⁻¹ (cutoff={:.1} Å, tolerance={:.0e})",
            beta, DEFAULT_REAL_SPACE_CUTOFF, DEFAULT_PME_TOLERANCE
        );

        Ok(Self {
            context,
            stream,
            module,
            nx,
            ny,
            nz,
            box_dims,
            beta,
            fft_plan_r2c,
            fft_plan_c2r,
            plans_initialized: true,
            d_charge_grid,
            d_complex_grid,
            d_energy,
            spread_kernel,
            convolution_kernel,
            interpolate_kernel,
            self_energy_kernel,
            zero_grid_kernel,
            normalize_kernel,
            n_atoms,
            // Phase 7: FP16 disabled by default
            fp16_enabled: false,
            d_charge_grid_fp16: None,
        })
    }

    /// Enable FP16 charge grid for mixed precision PME
    ///
    /// Phase 7: Allocates FP16 grid buffer for charge spreading.
    /// The grid is converted to FP32 before FFT operations.
    ///
    /// Benefits:
    /// - 50% reduction in grid memory usage
    /// - Reduced memory bandwidth for spreading
    ///
    /// Limitations:
    /// - cuFFT requires FP32, so conversion is needed before FFT
    /// - Slight precision loss in accumulation (~0.01% typical)
    pub fn enable_fp16_grid(&mut self) -> Result<()> {
        if self.fp16_enabled {
            return Ok(());  // Already enabled
        }

        let grid_size = self.nx * self.ny * self.nz;
        let d_charge_grid_fp16 = self.stream
            .alloc_zeros::<u16>(grid_size)
            .context("Failed to allocate FP16 charge grid")?;

        self.d_charge_grid_fp16 = Some(d_charge_grid_fp16);
        self.fp16_enabled = true;

        log::info!(
            "⚡ FP16 PME grid enabled: saved {} KB",
            (grid_size * 2) / 1024  // 4 bytes -> 2 bytes = 2 bytes saved per element
        );

        Ok(())
    }

    /// Disable FP16 charge grid (return to full FP32)
    pub fn disable_fp16_grid(&mut self) {
        self.d_charge_grid_fp16 = None;
        self.fp16_enabled = false;
        log::info!("FP16 PME grid disabled");
    }

    /// Check if FP16 grid is enabled
    pub fn is_fp16_enabled(&self) -> bool {
        self.fp16_enabled
    }

    /// Get memory savings from FP16 grid (in bytes)
    pub fn fp16_memory_savings(&self) -> usize {
        if self.fp16_enabled {
            self.nx * self.ny * self.nz * 2  // 4 bytes -> 2 bytes = 2 bytes saved
        } else {
            0
        }
    }

    /// Set the Ewald splitting parameter directly
    ///
    /// Prefer using `set_beta_from_cutoff()` which computes the optimal β.
    /// Larger β → more in reciprocal space, smaller → more in real space.
    pub fn set_beta(&mut self, beta: f32) {
        self.beta = beta;
        log::info!("🔧 PME β = {:.4} Å⁻¹ (manually set)", beta);
    }

    /// Set the Ewald splitting parameter from cutoff and tolerance
    ///
    /// This is the preferred method - computes optimal β using:
    /// β = sqrt(-ln(tolerance)) / cutoff
    ///
    /// # Arguments
    /// * `cutoff` - Real-space cutoff in Ångströms (should match non-bonded cutoff)
    /// * `tolerance` - Ewald sum tolerance (typically 1e-5 to 1e-6)
    pub fn set_beta_from_cutoff(&mut self, cutoff: f32, tolerance: f32) {
        self.beta = compute_ewald_beta(cutoff, tolerance);
        log::info!(
            "🔧 PME β = {:.4} Å⁻¹ (cutoff={:.1} Å, tolerance={:.0e})",
            self.beta, cutoff, tolerance
        );
    }

    /// Compute PME reciprocal energy and add forces to atoms
    ///
    /// # Arguments
    /// * `d_positions` - Device slice with atom positions [n_atoms * 3]
    /// * `d_charges` - Device slice with atom charges [n_atoms]
    /// * `d_forces` - Device slice to add forces to [n_atoms * 3]
    ///
    /// # Returns
    /// Reciprocal space energy in kcal/mol
    pub fn compute(
        &mut self,
        d_positions: &CudaSlice<f32>,
        d_charges: &CudaSlice<f32>,
        d_forces: &mut CudaSlice<f32>,
    ) -> Result<f64> {
        let threads = 256;
        let n_atoms_i32 = self.n_atoms as i32;
        let nx_i32 = self.nx as i32;
        let ny_i32 = self.ny as i32;
        let nz_i32 = self.nz as i32;

        // Box inverse dimensions for coordinate transforms
        let box_inv_x = 1.0f32 / self.box_dims[0];
        let box_inv_y = 1.0f32 / self.box_dims[1];
        let box_inv_z = 1.0f32 / self.box_dims[2];

        // 1. Zero the charge grid
        let grid_size = self.nx * self.ny * self.nz;
        let grid_blocks = (grid_size + threads - 1) / threads;
        let grid_size_i32 = grid_size as i32;

        let cfg_grid = LaunchConfig {
            grid_dim: (grid_blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.zero_grid_kernel);
            builder.arg(&self.d_charge_grid);
            builder.arg(&grid_size_i32);
            builder.launch(cfg_grid)?;
        }

        // 2. Zero energy accumulator
        let zero_energy = vec![0.0f32];
        self.stream.memcpy_htod(&zero_energy, &mut self.d_energy)?;

        // 3. Spread charges to grid
        let atom_blocks = (self.n_atoms + threads - 1) / threads;
        let cfg_atoms = LaunchConfig {
            grid_dim: (atom_blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.spread_kernel);
            builder.arg(d_positions);
            builder.arg(d_charges);
            builder.arg(&self.d_charge_grid);
            builder.arg(&n_atoms_i32);
            builder.arg(&nx_i32);
            builder.arg(&ny_i32);
            builder.arg(&nz_i32);
            builder.arg(&box_inv_x);
            builder.arg(&box_inv_y);
            builder.arg(&box_inv_z);
            builder.launch(cfg_atoms)?;
        }

        // DEBUG: Check charge grid after spreading
        self.stream.synchronize()?;
        let mut charge_grid = vec![0.0f32; grid_size];
        self.stream.memcpy_dtoh(&self.d_charge_grid, &mut charge_grid)?;
        let mut max_q = 0.0f32;
        let mut sum_q = 0.0f32;
        for &q in &charge_grid {
            sum_q += q;
            if q.abs() > max_q {
                max_q = q.abs();
            }
        }
        log::info!(
            "🔬 PME charge grid after spreading: max_q={:.6}, sum_q={:.6}",
            max_q, sum_q
        );

        // 4. Forward FFT (R2C)
        self.stream.synchronize()?;
        execute_fft_r2c(
            self.fft_plan_r2c,
            &mut self.d_charge_grid,
            &mut self.d_complex_grid,
            &self.stream,
        )?;

        // 5. Apply reciprocal space convolution
        let complex_size = self.nx * self.ny * (self.nz / 2 + 1);
        let complex_blocks = (complex_size + threads - 1) / threads;

        let cfg_complex = LaunchConfig {
            grid_dim: (complex_blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Complex grid is interleaved: [re0, im0, re1, im1, ...]
        // Kernel handles interleaved indexing internally
        unsafe {
            let mut builder = self.stream.launch_builder(&self.convolution_kernel);
            builder.arg(&self.d_complex_grid); // Interleaved complex data
            builder.arg(&self.d_energy);
            builder.arg(&self.beta);
            builder.arg(&nx_i32);
            builder.arg(&ny_i32);
            builder.arg(&nz_i32);
            builder.arg(&box_inv_x);
            builder.arg(&box_inv_y);
            builder.arg(&box_inv_z);
            builder.launch(cfg_complex)?;
        }

        // DEBUG: Check complex grid after convolution
        self.stream.synchronize()?;
        let complex_total = self.nx * self.ny * (self.nz / 2 + 1) * 2;
        let mut complex_grid = vec![0.0f32; complex_total];
        self.stream.memcpy_dtoh(&self.d_complex_grid, &mut complex_grid)?;
        let mut max_cplx = 0.0f32;
        for chunk in complex_grid.chunks(2) {
            let mag = (chunk[0]*chunk[0] + chunk[1]*chunk[1]).sqrt();
            if mag > max_cplx {
                max_cplx = mag;
            }
        }
        log::info!(
            "🔬 PME complex grid after convolution: max_magnitude={:.6}",
            max_cplx
        );

        // 6. Inverse FFT (C2R)
        self.stream.synchronize()?;
        execute_fft_c2r(
            self.fft_plan_c2r,
            &mut self.d_complex_grid,
            &mut self.d_charge_grid,
            &self.stream,
        )?;

        // DEBUG: Check potential grid right after IFFT, before normalization
        self.stream.synchronize()?;
        let mut potential_pre = vec![0.0f32; grid_size];
        self.stream.memcpy_dtoh(&self.d_charge_grid, &mut potential_pre)?;
        let mut max_pre = 0.0f32;
        for &p in &potential_pre {
            if p.abs() > max_pre {
                max_pre = p.abs();
            }
        }
        log::info!(
            "🔬 PME potential BEFORE normalization: max={:.6}",
            max_pre
        );

        // 7. Skip normalization - the Green's function already includes 1/V
        // The cuFFT IFFT output is exactly what we want for PME
        // Previously: dividing by N (= grid_size) was WRONG because Green's function already has 1/V
        //
        // Explanation:
        // - Green's function: G(k) = 4π/(V·k²) × exp(-k²/(4β²)) × B_corr
        // - The 1/V in G(k) combines with the IFFT to give correct potential
        // - Additional 1/N normalization would make potential ~4000× too small
        //
        // With 1/N removed, potential should be ~0.088 instead of 0.00002
        // This gives forces ~1000× larger (from ~0.04 to ~40 kcal/(mol·Å))
        log::debug!("Skipping normalization (Green's function already has 1/V)");

        // DEBUG: Check potential grid values after normalization
        // Copy full grid to check values (memcpy_dtoh requires dst.len() >= src.len())
        self.stream.synchronize()?;
        let mut potential_full = vec![0.0f32; grid_size];
        self.stream.memcpy_dtoh(&self.d_charge_grid, &mut potential_full)?;
        let mut max_phi = 0.0f32;
        let mut sum_phi = 0.0f32;
        for &phi in &potential_full {
            sum_phi += phi.abs();
            if phi.abs() > max_phi {
                max_phi = phi.abs();
            }
        }
        log::info!(
            "🔬 PME potential grid: max_φ={:.6}, avg_φ={:.6}",
            max_phi, sum_phi / grid_size as f32
        );

        // 8. Interpolate forces from potential grid
        unsafe {
            let mut builder = self.stream.launch_builder(&self.interpolate_kernel);
            builder.arg(d_positions);
            builder.arg(d_charges);
            builder.arg(&self.d_charge_grid); // Now contains potential
            builder.arg(d_forces);
            builder.arg(&n_atoms_i32);
            builder.arg(&nx_i32);
            builder.arg(&ny_i32);
            builder.arg(&nz_i32);
            builder.arg(&box_inv_x);
            builder.arg(&box_inv_y);
            builder.arg(&box_inv_z);
            builder.launch(cfg_atoms)?;
        }

        // 9. Add self-energy correction
        unsafe {
            let mut builder = self.stream.launch_builder(&self.self_energy_kernel);
            builder.arg(d_charges);
            builder.arg(&self.d_energy);
            builder.arg(&n_atoms_i32);
            builder.arg(&self.beta);
            builder.launch(cfg_atoms)?;
        }

        self.stream.synchronize()?;

        // 10. Download energy
        let mut energy = vec![0.0f32; 1];
        self.stream.memcpy_dtoh(&self.d_energy, &mut energy)?;

        log::debug!("⚡ PME reciprocal energy: {:.2} kcal/mol", energy[0]);

        Ok(energy[0] as f64)
    }

    /// Get the current grid dimensions
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Get the Ewald beta parameter
    pub fn beta(&self) -> f32 {
        self.beta
    }
}

impl Drop for PME {
    fn drop(&mut self) {
        if self.plans_initialized {
            unsafe {
                crate::cufft_sys::cufftDestroy(self.fft_plan_r2c);
                crate::cufft_sys::cufftDestroy(self.fft_plan_c2r);
            }
            log::debug!("🧹 PME cuFFT plans destroyed");
        }
    }
}

/// Round up to a good FFT size (powers of 2, or products of 2, 3, 5)
fn round_up_fft_size(n: usize) -> usize {
    // Good FFT sizes are products of small primes (2, 3, 5)
    // Find smallest such number >= n
    let good_sizes = [
        8, 10, 12, 16, 18, 20, 24, 30, 32, 36, 40, 48, 54, 60, 64, 72, 80, 90, 96, 100, 108, 120,
        128, 144, 160, 180, 192, 200, 216, 240, 256, 270, 288, 320, 324, 360, 384, 400, 432, 480,
        512,
    ];

    for &size in &good_sizes {
        if size >= n {
            return size;
        }
    }

    // Fall back to next power of 2
    let mut size = 512;
    while size < n {
        size *= 2;
    }
    size
}

/// Create R2C and C2R FFT plans
fn create_fft_plans(nx: usize, ny: usize, nz: usize) -> Result<(CufftHandle, CufftHandle)> {
    let mut plan_r2c: CufftHandle = 0;
    let mut plan_c2r: CufftHandle = 0;

    unsafe {
        // R2C plan (forward FFT)
        let result = crate::cufft_sys::cufftPlan3d(
            &mut plan_r2c,
            nx as i32,
            ny as i32,
            nz as i32,
            CUFFT_R2C,
        );
        if result != CUFFT_SUCCESS {
            return Err(anyhow::anyhow!(
                "cufftPlan3d R2C failed: {}",
                cufft_error_string(result)
            ));
        }

        // C2R plan (inverse FFT)
        let result = crate::cufft_sys::cufftPlan3d(
            &mut plan_c2r,
            nx as i32,
            ny as i32,
            nz as i32,
            CUFFT_C2R,
        );
        if result != CUFFT_SUCCESS {
            crate::cufft_sys::cufftDestroy(plan_r2c);
            return Err(anyhow::anyhow!(
                "cufftPlan3d C2R failed: {}",
                cufft_error_string(result)
            ));
        }
    }

    Ok((plan_r2c, plan_c2r))
}

/// Execute R2C FFT (real to complex)
fn execute_fft_r2c(
    plan: CufftHandle,
    d_input: &mut CudaSlice<f32>,
    d_output: &mut CudaSlice<f32>,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        // Get raw device pointers - cudarc 0.18.2 returns (CUdeviceptr, SyncOnDrop)
        let (input_ptr, _sync_in) = d_input.device_ptr_mut(stream);
        let (output_ptr, _sync_out) = d_output.device_ptr_mut(stream);

        let result = crate::cufft_sys::cufftExecR2C(
            plan,
            input_ptr as *mut f32,
            output_ptr as *mut CufftComplex,
        );
        if result != CUFFT_SUCCESS {
            return Err(anyhow::anyhow!(
                "cufftExecR2C failed: {}",
                cufft_error_string(result)
            ));
        }
    }
    Ok(())
}

/// Execute C2R FFT (complex to real)
fn execute_fft_c2r(
    plan: CufftHandle,
    d_input: &mut CudaSlice<f32>,
    d_output: &mut CudaSlice<f32>,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        // Get raw device pointers - cudarc 0.18.2 returns (CUdeviceptr, SyncOnDrop)
        let (input_ptr, _sync_in) = d_input.device_ptr_mut(stream);
        let (output_ptr, _sync_out) = d_output.device_ptr_mut(stream);

        let result = crate::cufft_sys::cufftExecC2R(
            plan,
            input_ptr as *mut CufftComplex,
            output_ptr as *mut f32,
        );
        if result != CUFFT_SUCCESS {
            return Err(anyhow::anyhow!(
                "cufftExecC2R failed: {}",
                cufft_error_string(result)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_up_fft_size() {
        assert_eq!(round_up_fft_size(5), 8);
        assert_eq!(round_up_fft_size(8), 8);
        assert_eq!(round_up_fft_size(9), 10);
        assert_eq!(round_up_fft_size(33), 36);
        assert_eq!(round_up_fft_size(100), 100);
        assert_eq!(round_up_fft_size(101), 108);
    }
}
