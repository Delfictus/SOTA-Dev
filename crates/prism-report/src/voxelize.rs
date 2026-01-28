//! Post-Run Voxelization Module
//!
//! Gaussian deposition from sparse event-cloud to dense MRC volumes.
//!
//! STRICT RULE: This runs ONLY post-run, never in the GPU stepping loop.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::event_cloud::{AblationPhase, EventCloud, PocketEvent};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Gaussian sigma for deposition (Å)
pub const GAUSSIAN_SIGMA: f32 = 1.5;

/// Cutoff radius = 3 * sigma (Å)
pub const GAUSSIAN_RADIUS: f32 = GAUSSIAN_SIGMA * 3.0;

/// Default voxel spacing (Å)
pub const DEFAULT_SPACING: f32 = 1.0;

/// Padding around bounding box (Å)
pub const DEFAULT_PADDING: f32 = 5.0;

// =============================================================================
// VOXEL GRID
// =============================================================================

/// Dense voxel grid for post-run rasterization
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    /// Grid data (X-major ordering: data[z * ny * nx + y * nx + x])
    pub data: Vec<f32>,

    /// Grid dimensions [nx, ny, nz]
    pub dims: [usize; 3],

    /// Grid origin (Å) - minimum corner
    pub origin: [f32; 3],

    /// Voxel spacing (Å)
    pub spacing: f32,
}

impl VoxelGrid {
    /// Create new zero-initialized grid
    pub fn new(dims: [usize; 3], origin: [f32; 3], spacing: f32) -> Self {
        let size = dims[0] * dims[1] * dims[2];
        Self {
            data: vec![0.0; size],
            dims,
            origin,
            spacing,
        }
    }

    /// Create grid from bounding box with padding
    pub fn from_bounds(min: [f32; 3], max: [f32; 3], spacing: f32, padding: f32) -> Self {
        let origin = [
            min[0] - padding,
            min[1] - padding,
            min[2] - padding,
        ];

        let dims = [
            ((max[0] - min[0] + 2.0 * padding) / spacing).ceil() as usize + 1,
            ((max[1] - min[1] + 2.0 * padding) / spacing).ceil() as usize + 1,
            ((max[2] - min[2] + 2.0 * padding) / spacing).ceil() as usize + 1,
        ];

        Self::new(dims, origin, spacing)
    }

    /// Get linear index from 3D coordinates
    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.dims[1] * self.dims[0] + y * self.dims[0] + x
    }

    /// Get value at voxel coordinates
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[self.index(x, y, z)]
    }

    /// Set value at voxel coordinates
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: f32) {
        let idx = self.index(x, y, z);
        self.data[idx] = value;
    }

    /// Add value at voxel coordinates
    #[inline]
    pub fn add(&mut self, x: usize, y: usize, z: usize, value: f32) {
        let idx = self.index(x, y, z);
        self.data[idx] += value;
    }

    /// Convert world coordinates to voxel coordinates
    pub fn world_to_voxel(&self, pos: [f32; 3]) -> [f32; 3] {
        [
            (pos[0] - self.origin[0]) / self.spacing,
            (pos[1] - self.origin[1]) / self.spacing,
            (pos[2] - self.origin[2]) / self.spacing,
        ]
    }

    /// Convert voxel coordinates to world coordinates
    pub fn voxel_to_world(&self, voxel: [usize; 3]) -> [f32; 3] {
        [
            self.origin[0] + voxel[0] as f32 * self.spacing,
            self.origin[1] + voxel[1] as f32 * self.spacing,
            self.origin[2] + voxel[2] as f32 * self.spacing,
        ]
    }

    /// Maximum value in grid
    pub fn max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Minimum value in grid
    pub fn min(&self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Mean value in grid
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// Normalize grid to [0, 1] range
    pub fn normalize(&mut self) {
        let min = self.min();
        let max = self.max();
        let range = max - min;

        if range > 1e-10 {
            for v in &mut self.data {
                *v = (*v - min) / range;
            }
        }
    }
}

// =============================================================================
// GAUSSIAN DEPOSITION
// =============================================================================

/// Deposit a single Gaussian at a position
pub fn deposit_gaussian(grid: &mut VoxelGrid, center: [f32; 3], amplitude: f32, sigma: f32) {
    let radius_voxels = (GAUSSIAN_RADIUS / grid.spacing).ceil() as i32;
    let voxel_center = grid.world_to_voxel(center);

    let cx = voxel_center[0] as i32;
    let cy = voxel_center[1] as i32;
    let cz = voxel_center[2] as i32;

    let sigma_sq = sigma * sigma;
    let two_sigma_sq = 2.0 * sigma_sq;

    for dz in -radius_voxels..=radius_voxels {
        let z = cz + dz;
        if z < 0 || z >= grid.dims[2] as i32 {
            continue;
        }

        for dy in -radius_voxels..=radius_voxels {
            let y = cy + dy;
            if y < 0 || y >= grid.dims[1] as i32 {
                continue;
            }

            for dx in -radius_voxels..=radius_voxels {
                let x = cx + dx;
                if x < 0 || x >= grid.dims[0] as i32 {
                    continue;
                }

                // Distance in world coordinates
                let world_pos = grid.voxel_to_world([x as usize, y as usize, z as usize]);
                let dist_sq = (world_pos[0] - center[0]).powi(2)
                    + (world_pos[1] - center[1]).powi(2)
                    + (world_pos[2] - center[2]).powi(2);

                // Gaussian kernel
                let value = amplitude * (-dist_sq / two_sigma_sq).exp();

                grid.add(x as usize, y as usize, z as usize, value);
            }
        }
    }
}

/// Deposit events from event cloud to grid
pub fn deposit_events(
    grid: &mut VoxelGrid,
    events: &[&PocketEvent],
    sigma: f32,
    weight_by_confidence: bool,
) {
    for event in events {
        let amplitude = if weight_by_confidence {
            event.confidence.max(0.1) // Minimum weight to avoid zero
        } else {
            1.0
        };

        deposit_gaussian(grid, event.center_xyz, amplitude, sigma);
    }
}

// =============================================================================
// VOXELIZER
// =============================================================================

/// Post-run voxelizer for event clouds
pub struct Voxelizer {
    /// Gaussian sigma (Å)
    pub sigma: f32,

    /// Voxel spacing (Å)
    pub spacing: f32,

    /// Padding around bounding box (Å)
    pub padding: f32,

    /// Weight by confidence
    pub weight_by_confidence: bool,
}

impl Default for Voxelizer {
    fn default() -> Self {
        Self {
            sigma: GAUSSIAN_SIGMA,
            spacing: DEFAULT_SPACING,
            padding: DEFAULT_PADDING,
            weight_by_confidence: true,
        }
    }
}

impl Voxelizer {
    /// Create new voxelizer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create voxelizer with custom sigma
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma;
        self
    }

    /// Create voxelizer with custom spacing
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// Voxelize entire event cloud
    pub fn voxelize(&self, cloud: &EventCloud) -> Option<VoxelGrid> {
        let (min, max) = cloud.bounding_box()?;
        let mut grid = VoxelGrid::from_bounds(min, max, self.spacing, self.padding);

        let all_events: Vec<&PocketEvent> = cloud.events.iter().collect();
        deposit_events(&mut grid, &all_events, self.sigma, self.weight_by_confidence);

        Some(grid)
    }

    /// Voxelize events for a specific phase
    pub fn voxelize_phase(&self, cloud: &EventCloud, phase: AblationPhase) -> Option<VoxelGrid> {
        let (min, max) = cloud.bounding_box()?;
        let mut grid = VoxelGrid::from_bounds(min, max, self.spacing, self.padding);

        let events = cloud.filter_phase(phase);
        deposit_events(&mut grid, &events, self.sigma, self.weight_by_confidence);

        Some(grid)
    }

    /// Generate occupancy map (normalized event density)
    pub fn occupancy_map(&self, cloud: &EventCloud) -> Option<VoxelGrid> {
        let mut grid = self.voxelize(cloud)?;
        grid.normalize();
        Some(grid)
    }

    /// Generate pocket field map (phase-combined)
    pub fn pocket_field_map(&self, cloud: &EventCloud) -> Option<VoxelGrid> {
        let (min, max) = cloud.bounding_box()?;
        let mut grid = VoxelGrid::from_bounds(min, max, self.spacing, self.padding);

        // Weight by phase: cryo_uv > cryo_only > baseline
        for event in &cloud.events {
            let phase_weight = match event.phase {
                AblationPhase::Baseline => 0.5,
                AblationPhase::CryoOnly => 0.75,
                AblationPhase::CryoUv => 1.0,
            };

            let amplitude = if self.weight_by_confidence {
                phase_weight * event.confidence.max(0.1)
            } else {
                phase_weight
            };

            deposit_gaussian(&mut grid, event.center_xyz, amplitude, self.sigma);
        }

        Some(grid)
    }
}

// =============================================================================
// MRC FILE FORMAT
// =============================================================================

/// MRC file header (2014 format)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct MrcHeader {
    /// Number of columns (X)
    nx: i32,
    /// Number of rows (Y)
    ny: i32,
    /// Number of sections (Z)
    nz: i32,
    /// Data type (2 = float32)
    mode: i32,
    /// Start of sub-volume (X)
    nxstart: i32,
    /// Start of sub-volume (Y)
    nystart: i32,
    /// Start of sub-volume (Z)
    nzstart: i32,
    /// Grid intervals (X)
    mx: i32,
    /// Grid intervals (Y)
    my: i32,
    /// Grid intervals (Z)
    mz: i32,
    /// Cell dimensions (Å) - X
    cella_x: f32,
    /// Cell dimensions (Å) - Y
    cella_y: f32,
    /// Cell dimensions (Å) - Z
    cella_z: f32,
    /// Cell angles - alpha
    cellb_alpha: f32,
    /// Cell angles - beta
    cellb_beta: f32,
    /// Cell angles - gamma
    cellb_gamma: f32,
    /// Axis mapping for columns
    mapc: i32,
    /// Axis mapping for rows
    mapr: i32,
    /// Axis mapping for sections
    maps: i32,
    /// Minimum density value
    dmin: f32,
    /// Maximum density value
    dmax: f32,
    /// Mean density value
    dmean: f32,
    /// Space group (1 = P1)
    ispg: i32,
    /// Number of symmetry bytes
    nsymbt: i32,
    /// Extra space (up to word 25)
    extra: [i32; 25],
    /// Origin (X) - Å
    origin_x: f32,
    /// Origin (Y) - Å
    origin_y: f32,
    /// Origin (Z) - Å
    origin_z: f32,
    /// File stamp
    map: [u8; 4],
    /// Machine stamp
    machst: [u8; 4],
    /// RMS deviation
    rms: f32,
    /// Number of labels
    nlabl: i32,
    /// Labels (10 x 80 chars)
    label: [[u8; 80]; 10],
}

impl MrcHeader {
    fn new(grid: &VoxelGrid) -> Self {
        let mut header = Self {
            nx: grid.dims[0] as i32,
            ny: grid.dims[1] as i32,
            nz: grid.dims[2] as i32,
            mode: 2, // float32
            nxstart: 0,
            nystart: 0,
            nzstart: 0,
            mx: grid.dims[0] as i32,
            my: grid.dims[1] as i32,
            mz: grid.dims[2] as i32,
            cella_x: grid.dims[0] as f32 * grid.spacing,
            cella_y: grid.dims[1] as f32 * grid.spacing,
            cella_z: grid.dims[2] as f32 * grid.spacing,
            cellb_alpha: 90.0,
            cellb_beta: 90.0,
            cellb_gamma: 90.0,
            mapc: 1,
            mapr: 2,
            maps: 3,
            dmin: grid.min(),
            dmax: grid.max(),
            dmean: grid.mean(),
            ispg: 1,
            nsymbt: 0,
            extra: [0; 25],
            origin_x: grid.origin[0],
            origin_y: grid.origin[1],
            origin_z: grid.origin[2],
            map: *b"MAP ",
            machst: [0x44, 0x41, 0x00, 0x00], // Little-endian
            rms: 0.0,
            nlabl: 1,
            label: [[0; 80]; 10],
        };

        // Set label
        let label_str = b"PRISM4D voxelization output";
        header.label[0][..label_str.len()].copy_from_slice(label_str);

        // Calculate RMS
        let mean = header.dmean;
        let rms: f32 = (grid.data.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / grid.data.len() as f32)
            .sqrt();
        header.rms = rms;

        header
    }
}

/// Write grid to MRC file
pub fn write_mrc(grid: &VoxelGrid, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path)
        .with_context(|| format!("Failed to create MRC file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    // Write header
    let header = MrcHeader::new(grid);
    let header_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &header as *const MrcHeader as *const u8,
            std::mem::size_of::<MrcHeader>(),
        )
    };
    writer.write_all(header_bytes)?;

    // Pad to 1024 bytes (standard MRC header size)
    let header_size = std::mem::size_of::<MrcHeader>();
    if header_size < 1024 {
        let padding = vec![0u8; 1024 - header_size];
        writer.write_all(&padding)?;
    }

    // Write data (already in the right order for MRC)
    for &value in &grid.data {
        writer.write_all(&value.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

// =============================================================================
// VOXELIZATION RESULTS
// =============================================================================

/// Results from voxelization
#[derive(Debug, Clone)]
pub struct VoxelizationResult {
    /// Occupancy grid
    pub occupancy: VoxelGrid,

    /// Pocket field grid (phase-weighted)
    pub pocket_field: VoxelGrid,

    /// Grid dimensions
    pub dims: [usize; 3],

    /// Voxel spacing (Å)
    pub spacing: f32,

    /// Total voxels above threshold
    pub voxels_above_threshold: usize,

    /// Estimated total volume (Å³)
    pub total_volume: f32,
}

impl VoxelizationResult {
    /// Write MRC files to output directory
    pub fn write_mrc_files(&self, output_dir: impl AsRef<Path>) -> Result<()> {
        let volumes_dir = output_dir.as_ref().join("volumes");
        std::fs::create_dir_all(&volumes_dir)?;

        write_mrc(&self.occupancy, volumes_dir.join("occupancy.mrc"))?;
        write_mrc(&self.pocket_field, volumes_dir.join("pocket_fields.mrc"))?;

        Ok(())
    }
}

/// Voxelize event cloud and compute results
pub fn voxelize_event_cloud(cloud: &EventCloud, threshold: f32) -> Option<VoxelizationResult> {
    let voxelizer = Voxelizer::new();

    let occupancy = voxelizer.occupancy_map(cloud)?;
    let pocket_field = voxelizer.pocket_field_map(cloud)?;

    // Count voxels above threshold
    let voxels_above = occupancy.data.iter().filter(|&&v| v > threshold).count();
    let voxel_volume = voxelizer.spacing.powi(3);
    let total_volume = voxels_above as f32 * voxel_volume;

    Some(VoxelizationResult {
        dims: occupancy.dims,
        spacing: voxelizer.spacing,
        occupancy,
        pocket_field,
        voxels_above_threshold: voxels_above,
        total_volume,
    })
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_voxel_grid_basic() {
        let grid = VoxelGrid::new([10, 10, 10], [0.0, 0.0, 0.0], 1.0);
        assert_eq!(grid.dims, [10, 10, 10]);
        assert_eq!(grid.data.len(), 1000);
    }

    #[test]
    fn test_gaussian_deposit() {
        let mut grid = VoxelGrid::new([20, 20, 20], [-10.0, -10.0, -10.0], 1.0);
        deposit_gaussian(&mut grid, [0.0, 0.0, 0.0], 1.0, GAUSSIAN_SIGMA);

        // Center should have highest value
        let center_val = grid.get(10, 10, 10);
        let edge_val = grid.get(0, 0, 0);
        assert!(center_val > edge_val);
        assert!(center_val > 0.9); // Near 1.0 at center
    }

    #[test]
    fn test_mrc_write() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.mrc");

        let mut grid = VoxelGrid::new([10, 10, 10], [0.0, 0.0, 0.0], 1.0);
        grid.set(5, 5, 5, 1.0);

        write_mrc(&grid, &path).unwrap();
        assert!(path.exists());

        // Check file size: 1024 header + 10*10*10*4 bytes
        let metadata = std::fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), 1024 + 4000);
    }

    #[test]
    fn test_voxelizer() {
        use crate::event_cloud::{PocketEvent, TempPhase};

        let mut cloud = EventCloud::new();
        cloud.push(PocketEvent {
            center_xyz: [0.0, 0.0, 0.0],
            volume_a3: 100.0,
            spike_count: 5,
            phase: AblationPhase::CryoUv,
            temp_phase: TempPhase::Cold,
            replicate_id: 0,
            frame_idx: 0,
            residues: vec![],
            confidence: 0.8,
            wavelength_nm: Some(280.0),
        });
        cloud.push(PocketEvent {
            center_xyz: [5.0, 5.0, 5.0],
            volume_a3: 150.0,
            spike_count: 8,
            phase: AblationPhase::CryoOnly,
            temp_phase: TempPhase::Ramp,
            replicate_id: 0,
            frame_idx: 100,
            residues: vec![],
            confidence: 0.9,
            wavelength_nm: None,
        });

        let voxelizer = Voxelizer::new();
        let grid = voxelizer.voxelize(&cloud).unwrap();

        assert!(grid.max() > 0.0);
    }
}
