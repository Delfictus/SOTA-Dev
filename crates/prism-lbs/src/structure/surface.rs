//! Surface accessibility computation (CPU reference implementation)

use crate::LbsError;
#[cfg(feature = "cuda")]
use prism_gpu::context::GpuContext;
#[cfg(feature = "cuda")]
use prism_gpu::global_context::GlobalGpuContext;
#[cfg(feature = "cuda")]
use prism_gpu::LbsGpu;

use super::{distance, distance_squared, pdb_parser::ProteinStructure};

/// Computes solvent-accessible surface areas using a simple Shrake-Rupley style sampler
#[derive(Debug, Clone)]
pub struct SurfaceComputer {
    /// Probe radius in Å (default: 1.4 Å)
    pub probe_radius: f64,
    /// Number of sample points on the sphere
    pub samples: usize,
    /// Fraction of exposed points required to mark an atom as surface
    pub min_surface_fraction: f64,
}

impl Default for SurfaceComputer {
    fn default() -> Self {
        Self {
            probe_radius: 1.4,
            samples: 480,
            min_surface_fraction: 0.05,
        }
    }
}

impl SurfaceComputer {
    /// Compute SASA using GPU if available; fall back to CPU otherwise.
    #[cfg(feature = "cuda")]
    pub fn compute_gpu(
        &self,
        structure: &mut ProteinStructure,
        gpu_ctx: &GpuContext,
    ) -> Result<(), LbsError> {
        if structure.atoms.is_empty() {
            return Ok(());
        }
        let coords: Vec<[f32; 3]> = structure
            .atoms
            .iter()
            .map(|a| [a.coord[0] as f32, a.coord[1] as f32, a.coord[2] as f32])
            .collect();
        let radii: Vec<f32> = structure
            .atoms
            .iter()
            .map(|a| (a.vdw_radius() + self.probe_radius) as f32)
            .collect();

        // Try to use pre-loaded LbsGpu from GlobalGpuContext (zero PTX overhead)
        // Fall back to creating a new one if GlobalGpuContext isn't available
        let (sasa, surface) = if let Some(lbs_gpu) = GlobalGpuContext::try_get().ok().and_then(|g| g.lbs_locked()) {
            log::debug!("Using pre-loaded LbsGpu for surface computation (zero PTX overhead)");
            lbs_gpu.surface_accessibility(&coords, &radii, self.samples as i32, self.probe_radius as f32)
                .map_err(|e| LbsError::Gpu(format!("Surface kernel failed: {}", e)))?
        } else {
            log::debug!("GlobalGpuContext LbsGpu not available, creating new instance");
            let lbs_gpu = LbsGpu::new(gpu_ctx.device().clone(), &gpu_ctx.ptx_dir())
                .map_err(|e| LbsError::Gpu(format!("Failed to init LbsGpu: {}", e)))?;
            lbs_gpu.surface_accessibility(&coords, &radii, self.samples as i32, self.probe_radius as f32)
                .map_err(|e| LbsError::Gpu(format!("Surface kernel failed: {}", e)))?
        };

        let mut max_center_distance: f64 = 0.0;
        structure.recompute_geometry();
        for atom in &structure.atoms {
            let d = distance(&atom.coord, &structure.center_of_mass);
            if d > max_center_distance {
                max_center_distance = d;
            }
        }

        for (idx, atom) in structure.atoms.iter_mut().enumerate() {
            atom.sasa = sasa.get(idx).cloned().unwrap_or(0.0) as f64;
            atom.is_surface = surface.get(idx).cloned().unwrap_or(0) != 0;
            atom.depth =
                (max_center_distance - distance(&atom.coord, &structure.center_of_mass)).max(0.0);
            atom.curvature = if atom.sasa > 0.0 {
                1.0 - atom.sasa / (4.0 * std::f64::consts::PI * (radii[idx] as f64).powi(2))
            } else {
                1.0
            };
        }
        structure.refresh_residue_properties();
        Ok(())
    }

    /// Compute per-atom SASA and residue SASA, marking surface atoms in-place
    pub fn compute(&self, structure: &mut ProteinStructure) -> Result<(), LbsError> {
        if structure.atoms.is_empty() {
            return Ok(());
        }

        structure.recompute_geometry();

        let points = fibonacci_sphere(self.samples);
        let radii: Vec<f64> = structure
            .atoms
            .iter()
            .map(|a| a.vdw_radius() + self.probe_radius)
            .collect();
        let coords: Vec<[f64; 3]> = structure.atoms.iter().map(|a| a.coord).collect();

        // Pre-compute neighbor lists to avoid quadratic scans for every sample point
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); structure.atoms.len()];
        for i in 0..structure.atoms.len() {
            for j in (i + 1)..structure.atoms.len() {
                let cutoff = radii[i] + radii[j] + self.probe_radius;
                if distance_squared(&coords[i], &coords[j]) <= cutoff * cutoff {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
        }

        let center = structure.center_of_mass;
        let mut max_center_distance: f64 = 0.0;
        for coord in &coords {
            max_center_distance = max_center_distance.max(distance(coord, &center));
        }

        for (idx, atom) in structure.atoms.iter_mut().enumerate() {
            let radius = radii[idx];
            let mut exposed_points = 0usize;

            for p in &points {
                let sample = [
                    coords[idx][0] + p[0] * radius,
                    coords[idx][1] + p[1] * radius,
                    coords[idx][2] + p[2] * radius,
                ];

                let occluded = neighbors[idx].iter().any(|&n| {
                    let other_radius = radii[n];
                    distance_squared(&sample, &coords[n]) < other_radius * other_radius
                });

                if !occluded {
                    exposed_points += 1;
                }
            }

            let fraction_exposed = exposed_points as f64 / points.len() as f64;
            atom.sasa = 4.0 * std::f64::consts::PI * radius * radius * fraction_exposed;
            atom.is_surface = fraction_exposed >= self.min_surface_fraction;
            atom.depth = (max_center_distance - distance(&coords[idx], &center)).max(0.0);
            atom.curvature = 1.0 - fraction_exposed;
        }

        structure.refresh_residue_properties();
        Ok(())
    }
}

fn fibonacci_sphere(samples: usize) -> Vec<[f64; 3]> {
    let samples = samples.max(1);
    let golden_angle = std::f64::consts::PI * (3.0 - (5.0_f64).sqrt());

    (0..samples)
        .map(|i| {
            let y = 1.0 - (2.0 * i as f64) / (samples.saturating_sub(1).max(1)) as f64;
            let radius = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f64;

            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            [x, y, z]
        })
        .collect()
}
