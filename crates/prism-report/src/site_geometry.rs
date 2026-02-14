//! Site geometry computation: depth and mouth area from voxel grids
//!
//! Computes depth_proxy_A and mouth_area_proxy_A2 from post-run voxel grids.
//!
//! Algorithm:
//! 1. Extract site-local window (AABB around centroid)
//! 2. Build protein_solid mask (vdW + probe radius)
//! 3. Threshold density to get pocket_mask
//! 4. Flood fill from boundary to get bulk_solvent
//! 5. Identify mouth voxels (pocket adjacent to bulk_solvent)
//! 6. Multi-source BFS for depth computation
//! 7. Connected components for multiple openings

use std::collections::{HashSet, VecDeque};

/// Van der Waals radii for common elements (in Angstroms)
pub fn vdw_radius(element: &str) -> f32 {
    match element.to_uppercase().as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "F" => 1.47,
        "CL" => 1.75,
        "BR" => 1.85,
        "I" => 1.98,
        "FE" => 1.94,
        "ZN" => 1.39,
        "MG" => 1.73,
        "CA" => 1.97,
        "NA" => 2.27,
        "K" => 2.75,
        _ => 1.70, // Default to carbon
    }
}

/// Result of site geometry computation
#[derive(Debug, Clone, Default)]
pub struct SiteGeometryResult {
    /// Depth to mouth (p95 of distances), in Angstroms
    pub depth_proxy_pocket_a: Option<f32>,
    /// Depth to protein surface (p95), in Angstroms
    pub depth_proxy_surface_a: Option<f32>,
    /// Largest opening mouth area, in Angstroms^2
    pub mouth_area_proxy_a2: Option<f32>,
    /// Total mouth area (all openings), in Angstroms^2
    pub mouth_area_total_a2: Option<f32>,
    /// Number of distinct openings
    pub n_openings: Option<usize>,
}

/// Configuration for site geometry computation
#[derive(Debug, Clone)]
pub struct SiteGeometryConfig {
    /// Probe radius for solvent-accessible surface (Angstroms)
    pub probe_radius: f32,
    /// Minimum half-width for site window (Angstroms)
    pub min_half_width: f32,
    /// Maximum half-width for site window (Angstroms)
    pub max_half_width: f32,
    /// Padding beyond cluster radius (Angstroms)
    pub radius_padding: f32,
    /// Percentile for depth computation (0-100)
    pub depth_percentile: f32,
}

impl Default for SiteGeometryConfig {
    fn default() -> Self {
        Self {
            probe_radius: 1.4,
            min_half_width: 12.0,
            max_half_width: 32.0,
            radius_padding: 10.0,
            depth_percentile: 95.0,
        }
    }
}

/// 3D voxel grid for local computation
pub struct LocalGrid {
    /// Grid dimensions [nx, ny, nz]
    pub dims: [usize; 3],
    /// Voxel spacing (Angstroms)
    pub spacing: f32,
    /// Grid origin in world coordinates
    pub origin: [f32; 3],
    /// Voxel data (flattened, z-fastest)
    pub data: Vec<f32>,
}

impl LocalGrid {
    /// Create a new local grid
    pub fn new(dims: [usize; 3], spacing: f32, origin: [f32; 3]) -> Self {
        let size = dims[0] * dims[1] * dims[2];
        Self {
            dims,
            spacing,
            origin,
            data: vec![0.0; size],
        }
    }

    /// Convert world coordinates to voxel indices
    pub fn world_to_voxel(&self, pos: [f32; 3]) -> Option<[usize; 3]> {
        let vx = ((pos[0] - self.origin[0]) / self.spacing).floor() as i32;
        let vy = ((pos[1] - self.origin[1]) / self.spacing).floor() as i32;
        let vz = ((pos[2] - self.origin[2]) / self.spacing).floor() as i32;

        if vx >= 0
            && vy >= 0
            && vz >= 0
            && (vx as usize) < self.dims[0]
            && (vy as usize) < self.dims[1]
            && (vz as usize) < self.dims[2]
        {
            Some([vx as usize, vy as usize, vz as usize])
        } else {
            None
        }
    }

    /// Convert voxel indices to world coordinates (voxel center)
    pub fn voxel_to_world(&self, idx: [usize; 3]) -> [f32; 3] {
        [
            self.origin[0] + (idx[0] as f32 + 0.5) * self.spacing,
            self.origin[1] + (idx[1] as f32 + 0.5) * self.spacing,
            self.origin[2] + (idx[2] as f32 + 0.5) * self.spacing,
        ]
    }

    /// Get linear index from 3D indices
    #[inline]
    pub fn linear_idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    /// Get value at voxel
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[self.linear_idx(x, y, z)]
    }

    /// Set value at voxel
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: f32) {
        let idx = self.linear_idx(x, y, z);
        self.data[idx] = val;
    }

    /// Check if indices are valid
    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.dims[0]
            && (y as usize) < self.dims[1]
            && (z as usize) < self.dims[2]
    }
}

/// Boolean mask for voxels
pub struct BoolGrid {
    pub dims: [usize; 3],
    pub data: Vec<bool>,
}

impl BoolGrid {
    pub fn new(dims: [usize; 3], initial: bool) -> Self {
        let size = dims[0] * dims[1] * dims[2];
        Self {
            dims,
            data: vec![initial; size],
        }
    }

    #[inline]
    pub fn linear_idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> bool {
        self.data[self.linear_idx(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: bool) {
        let idx = self.linear_idx(x, y, z);
        self.data[idx] = val;
    }

    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.dims[0]
            && (y as usize) < self.dims[1]
            && (z as usize) < self.dims[2]
    }
}

/// 6-neighbor offsets for 3D grid traversal
const NEIGHBORS_6: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Compute site geometry from density grid and protein atoms
pub fn compute_site_geometry(
    density_grid: &LocalGrid,
    protein_atoms: &[([f32; 3], &str)], // (position, element)
    site_centroid: [f32; 3],
    cluster_radius: f32,
    config: &SiteGeometryConfig,
) -> SiteGeometryResult {
    // Step 1: Determine site window
    let half_width = (cluster_radius + config.radius_padding)
        .max(config.min_half_width)
        .min(config.max_half_width);

    let window_min = [
        site_centroid[0] - half_width,
        site_centroid[1] - half_width,
        site_centroid[2] - half_width,
    ];
    let window_max = [
        site_centroid[0] + half_width,
        site_centroid[1] + half_width,
        site_centroid[2] + half_width,
    ];

    // Convert to voxel indices
    let vmin = density_grid.world_to_voxel(window_min);
    let vmax = density_grid.world_to_voxel(window_max);

    let (vmin, vmax) = match (vmin, vmax) {
        (Some(min), Some(max)) => (min, max),
        _ => return SiteGeometryResult::default(), // Window outside grid
    };

    // Ensure valid window
    let window_dims = [
        (vmax[0].saturating_sub(vmin[0])).max(1),
        (vmax[1].saturating_sub(vmin[1])).max(1),
        (vmax[2].saturating_sub(vmin[2])).max(1),
    ];

    let window_origin = [
        density_grid.origin[0] + vmin[0] as f32 * density_grid.spacing,
        density_grid.origin[1] + vmin[1] as f32 * density_grid.spacing,
        density_grid.origin[2] + vmin[2] as f32 * density_grid.spacing,
    ];

    // Extract local density window
    let mut local_density = LocalGrid::new(window_dims, density_grid.spacing, window_origin);
    for z in 0..window_dims[2] {
        for y in 0..window_dims[1] {
            for x in 0..window_dims[0] {
                let gx = vmin[0] + x;
                let gy = vmin[1] + y;
                let gz = vmin[2] + z;
                if gx < density_grid.dims[0] && gy < density_grid.dims[1] && gz < density_grid.dims[2]
                {
                    local_density.set(x, y, z, density_grid.get(gx, gy, gz));
                }
            }
        }
    }

    // Step 2: Build protein_solid mask
    let protein_solid = build_protein_solid_mask(
        &local_density,
        protein_atoms,
        config.probe_radius,
    );

    // Step 3: Compute pocket threshold using robust statistics
    let t_pocket = compute_pocket_threshold(&local_density);
    if t_pocket <= 0.0 {
        return SiteGeometryResult::default();
    }

    // Step 4: Create pocket mask with threshold
    let mut pocket_mask = BoolGrid::new(window_dims, false);
    for z in 0..window_dims[2] {
        for y in 0..window_dims[1] {
            for x in 0..window_dims[0] {
                if local_density.get(x, y, z) >= t_pocket {
                    pocket_mask.set(x, y, z, true);
                }
            }
        }
    }

    // Step 5: Keep only pocket component containing voxel nearest to centroid
    let centroid_voxel = local_density.world_to_voxel(site_centroid);
    if let Some(cv) = centroid_voxel {
        pocket_mask = keep_connected_to_centroid(&pocket_mask, cv);
    }

    // Step 6: Flood fill to find bulk_solvent
    let bulk_solvent = compute_bulk_solvent(&protein_solid);

    // Step 7: Identify mouth voxels and faces
    let (mouth_voxels, solvent_faces) =
        compute_mouth_voxels_and_faces(&pocket_mask, &bulk_solvent);

    if mouth_voxels.is_empty() {
        return SiteGeometryResult::default();
    }

    // Step 8: Compute openings via connected components
    let (n_openings, opening_areas) =
        compute_openings(&mouth_voxels, &solvent_faces, &pocket_mask, local_density.spacing);

    let mouth_area_total = opening_areas.iter().sum::<f32>();
    let mouth_area_proxy = opening_areas.iter().cloned().fold(0.0f32, f32::max);

    // Step 9: Multi-source BFS for depth to mouth
    let depths_to_mouth = compute_depths_bfs(&pocket_mask, &mouth_voxels, local_density.spacing);

    let depth_proxy_pocket = if !depths_to_mouth.is_empty() {
        Some(percentile_nearest_rank(&depths_to_mouth, config.depth_percentile))
    } else {
        None
    };

    // Step 10: Compute distance transform for protein surface depth
    let depth_proxy_surface = compute_surface_depth(
        &pocket_mask,
        &protein_solid,
        local_density.spacing,
        config.depth_percentile,
    );

    SiteGeometryResult {
        depth_proxy_pocket_a: depth_proxy_pocket,
        depth_proxy_surface_a: depth_proxy_surface,
        mouth_area_proxy_a2: Some(mouth_area_proxy),
        mouth_area_total_a2: Some(mouth_area_total),
        n_openings: Some(n_openings),
    }
}

/// Build protein solid mask by stamping atom spheres
fn build_protein_solid_mask(
    grid: &LocalGrid,
    atoms: &[([f32; 3], &str)],
    probe_radius: f32,
) -> BoolGrid {
    let mut mask = BoolGrid::new(grid.dims, false);

    for (pos, element) in atoms {
        let radius = vdw_radius(element) + probe_radius;
        let radius_sq = radius * radius;

        // Find voxels within radius of atom
        let vmin = grid.world_to_voxel([
            pos[0] - radius - grid.spacing,
            pos[1] - radius - grid.spacing,
            pos[2] - radius - grid.spacing,
        ]);
        let vmax = grid.world_to_voxel([
            pos[0] + radius + grid.spacing,
            pos[1] + radius + grid.spacing,
            pos[2] + radius + grid.spacing,
        ]);

        let (vmin, vmax) = match (vmin, vmax) {
            (Some(min), Some(max)) => (min, max),
            _ => continue,
        };

        for z in vmin[2]..=vmax[2].min(grid.dims[2] - 1) {
            for y in vmin[1]..=vmax[1].min(grid.dims[1] - 1) {
                for x in vmin[0]..=vmax[0].min(grid.dims[0] - 1) {
                    let voxel_center = grid.voxel_to_world([x, y, z]);
                    let dx = voxel_center[0] - pos[0];
                    let dy = voxel_center[1] - pos[1];
                    let dz = voxel_center[2] - pos[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    if dist_sq <= radius_sq {
                        mask.set(x, y, z, true);
                    }
                }
            }
        }
    }

    mask
}

/// Compute pocket threshold using robust statistics
fn compute_pocket_threshold(grid: &LocalGrid) -> f32 {
    // Collect positive density values
    let positive: Vec<f32> = grid.data.iter().cloned().filter(|&v| v > 0.0).collect();

    if positive.is_empty() {
        return 0.0;
    }

    // Compute statistics
    let mean: f32 = positive.iter().sum::<f32>() / positive.len() as f32;
    let variance: f32 = positive.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / positive.len() as f32;
    let std = variance.sqrt();

    let p90 = percentile_nearest_rank(&positive, 90.0);

    // t_pocket = max(p90, mean + 0.75*std)
    let t_pocket = p90.max(mean + 0.75 * std);

    // Enforce floor
    t_pocket.max(1e-6)
}

/// Keep only the connected component containing the centroid voxel
fn keep_connected_to_centroid(mask: &BoolGrid, centroid: [usize; 3]) -> BoolGrid {
    let mut result = BoolGrid::new(mask.dims, false);

    // Find nearest pocket voxel to centroid
    let start = if mask.get(centroid[0], centroid[1], centroid[2]) {
        centroid
    } else {
        // Search for nearest pocket voxel
        let mut best_dist_sq = f32::MAX;
        let mut best = centroid;
        for z in 0..mask.dims[2] {
            for y in 0..mask.dims[1] {
                for x in 0..mask.dims[0] {
                    if mask.get(x, y, z) {
                        let dx = x as f32 - centroid[0] as f32;
                        let dy = y as f32 - centroid[1] as f32;
                        let dz = z as f32 - centroid[2] as f32;
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        if dist_sq < best_dist_sq {
                            best_dist_sq = dist_sq;
                            best = [x, y, z];
                        }
                    }
                }
            }
        }
        if best_dist_sq == f32::MAX {
            return result; // No pocket voxels
        }
        best
    };

    // BFS to find connected component
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    queue.push_back(start);
    visited.insert((start[0], start[1], start[2]));

    while let Some(v) = queue.pop_front() {
        result.set(v[0], v[1], v[2], true);

        for (dx, dy, dz) in NEIGHBORS_6 {
            let nx = v[0] as i32 + dx;
            let ny = v[1] as i32 + dy;
            let nz = v[2] as i32 + dz;

            if mask.in_bounds(nx, ny, nz) {
                let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                if mask.get(nx, ny, nz) && !visited.contains(&(nx, ny, nz)) {
                    visited.insert((nx, ny, nz));
                    queue.push_back([nx, ny, nz]);
                }
            }
        }
    }

    result
}

/// Compute bulk solvent by flood fill from boundary
fn compute_bulk_solvent(protein_solid: &BoolGrid) -> BoolGrid {
    let mut bulk = BoolGrid::new(protein_solid.dims, false);
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    // Seed from boundary voxels that are not protein
    let [nx, ny, nz] = protein_solid.dims;

    // X boundaries
    for y in 0..ny {
        for z in 0..nz {
            if !protein_solid.get(0, y, z) {
                queue.push_back([0, y, z]);
                visited.insert((0, y, z));
            }
            if !protein_solid.get(nx - 1, y, z) {
                queue.push_back([nx - 1, y, z]);
                visited.insert((nx - 1, y, z));
            }
        }
    }

    // Y boundaries
    for x in 0..nx {
        for z in 0..nz {
            if !protein_solid.get(x, 0, z) && !visited.contains(&(x, 0, z)) {
                queue.push_back([x, 0, z]);
                visited.insert((x, 0, z));
            }
            if !protein_solid.get(x, ny - 1, z) && !visited.contains(&(x, ny - 1, z)) {
                queue.push_back([x, ny - 1, z]);
                visited.insert((x, ny - 1, z));
            }
        }
    }

    // Z boundaries
    for x in 0..nx {
        for y in 0..ny {
            if !protein_solid.get(x, y, 0) && !visited.contains(&(x, y, 0)) {
                queue.push_back([x, y, 0]);
                visited.insert((x, y, 0));
            }
            if !protein_solid.get(x, y, nz - 1) && !visited.contains(&(x, y, nz - 1)) {
                queue.push_back([x, y, nz - 1]);
                visited.insert((x, y, nz - 1));
            }
        }
    }

    // Flood fill
    while let Some(v) = queue.pop_front() {
        bulk.set(v[0], v[1], v[2], true);

        for (dx, dy, dz) in NEIGHBORS_6 {
            let nx = v[0] as i32 + dx;
            let ny = v[1] as i32 + dy;
            let nz = v[2] as i32 + dz;

            if bulk.in_bounds(nx, ny, nz) {
                let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                if !protein_solid.get(nx, ny, nz) && !visited.contains(&(nx, ny, nz)) {
                    visited.insert((nx, ny, nz));
                    queue.push_back([nx, ny, nz]);
                }
            }
        }
    }

    bulk
}

/// Compute mouth voxels and count solvent-facing faces
fn compute_mouth_voxels_and_faces(
    pocket: &BoolGrid,
    bulk_solvent: &BoolGrid,
) -> (Vec<[usize; 3]>, Vec<usize>) {
    let mut mouth_voxels = Vec::new();
    let mut solvent_faces = Vec::new(); // Number of solvent-facing faces per mouth voxel

    for z in 0..pocket.dims[2] {
        for y in 0..pocket.dims[1] {
            for x in 0..pocket.dims[0] {
                if !pocket.get(x, y, z) {
                    continue;
                }

                let mut face_count = 0;
                for (dx, dy, dz) in NEIGHBORS_6 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;

                    if bulk_solvent.in_bounds(nx, ny, nz) {
                        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                        if bulk_solvent.get(nx, ny, nz) {
                            face_count += 1;
                        }
                    }
                }

                if face_count > 0 {
                    mouth_voxels.push([x, y, z]);
                    solvent_faces.push(face_count);
                }
            }
        }
    }

    (mouth_voxels, solvent_faces)
}

/// Compute connected components of mouth voxels (openings)
fn compute_openings(
    mouth_voxels: &[[usize; 3]],
    solvent_faces: &[usize],
    pocket: &BoolGrid,
    spacing: f32,
) -> (usize, Vec<f32>) {
    if mouth_voxels.is_empty() {
        return (0, vec![]);
    }

    // Build mouth voxel set for quick lookup
    let mouth_set: HashSet<(usize, usize, usize)> = mouth_voxels
        .iter()
        .map(|v| (v[0], v[1], v[2]))
        .collect();

    // Map from voxel to index in mouth_voxels
    let voxel_to_idx: std::collections::HashMap<(usize, usize, usize), usize> = mouth_voxels
        .iter()
        .enumerate()
        .map(|(i, v)| ((v[0], v[1], v[2]), i))
        .collect();

    let mut visited = vec![false; mouth_voxels.len()];
    let mut opening_areas = Vec::new();
    let face_area = spacing * spacing;

    for start_idx in 0..mouth_voxels.len() {
        if visited[start_idx] {
            continue;
        }

        // BFS to find connected component
        let mut queue = VecDeque::new();
        queue.push_back(start_idx);
        visited[start_idx] = true;

        let mut component_face_count = 0usize;

        while let Some(idx) = queue.pop_front() {
            component_face_count += solvent_faces[idx];

            let v = mouth_voxels[idx];
            for (dx, dy, dz) in NEIGHBORS_6 {
                let nx = v[0] as i32 + dx;
                let ny = v[1] as i32 + dy;
                let nz = v[2] as i32 + dz;

                if pocket.in_bounds(nx, ny, nz) {
                    let key = (nx as usize, ny as usize, nz as usize);
                    if let Some(&neighbor_idx) = voxel_to_idx.get(&key) {
                        if !visited[neighbor_idx] {
                            visited[neighbor_idx] = true;
                            queue.push_back(neighbor_idx);
                        }
                    }
                }
            }
        }

        opening_areas.push(component_face_count as f32 * face_area);
    }

    (opening_areas.len(), opening_areas)
}

/// Multi-source BFS to compute distances from mouth voxels
fn compute_depths_bfs(pocket: &BoolGrid, mouth_voxels: &[[usize; 3]], spacing: f32) -> Vec<f32> {
    let mut distances = vec![f32::MAX; pocket.data.len()];
    let mut queue = VecDeque::new();

    // Initialize mouth voxels with distance 0
    for v in mouth_voxels {
        let idx = pocket.linear_idx(v[0], v[1], v[2]);
        distances[idx] = 0.0;
        queue.push_back(*v);
    }

    // BFS
    while let Some(v) = queue.pop_front() {
        let current_dist = distances[pocket.linear_idx(v[0], v[1], v[2])];

        for (dx, dy, dz) in NEIGHBORS_6 {
            let nx = v[0] as i32 + dx;
            let ny = v[1] as i32 + dy;
            let nz = v[2] as i32 + dz;

            if pocket.in_bounds(nx, ny, nz) {
                let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                if pocket.get(nx, ny, nz) {
                    let neighbor_idx = pocket.linear_idx(nx, ny, nz);
                    let new_dist = current_dist + spacing;
                    if new_dist < distances[neighbor_idx] {
                        distances[neighbor_idx] = new_dist;
                        queue.push_back([nx, ny, nz]);
                    }
                }
            }
        }
    }

    // Collect distances for pocket voxels
    let mut pocket_distances = Vec::new();
    for z in 0..pocket.dims[2] {
        for y in 0..pocket.dims[1] {
            for x in 0..pocket.dims[0] {
                if pocket.get(x, y, z) {
                    let d = distances[pocket.linear_idx(x, y, z)];
                    if d < f32::MAX {
                        pocket_distances.push(d);
                    }
                }
            }
        }
    }

    pocket_distances
}

/// Compute depth to protein surface using distance transform
fn compute_surface_depth(
    pocket: &BoolGrid,
    protein_solid: &BoolGrid,
    spacing: f32,
    percentile: f32,
) -> Option<f32> {
    // Find protein surface voxels (protein voxels adjacent to non-protein)
    let mut surface_voxels = Vec::new();
    for z in 0..protein_solid.dims[2] {
        for y in 0..protein_solid.dims[1] {
            for x in 0..protein_solid.dims[0] {
                if !protein_solid.get(x, y, z) {
                    continue;
                }

                let mut is_surface = false;
                for (dx, dy, dz) in NEIGHBORS_6 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;

                    if !protein_solid.in_bounds(nx, ny, nz) {
                        is_surface = true;
                        break;
                    }
                    let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                    if !protein_solid.get(nx, ny, nz) {
                        is_surface = true;
                        break;
                    }
                }

                if is_surface {
                    surface_voxels.push([x, y, z]);
                }
            }
        }
    }

    if surface_voxels.is_empty() {
        return None;
    }

    // BFS from surface voxels to compute distance transform
    let mut distances = vec![f32::MAX; pocket.data.len()];
    let mut queue = VecDeque::new();

    for v in &surface_voxels {
        let idx = pocket.linear_idx(v[0], v[1], v[2]);
        distances[idx] = 0.0;
        queue.push_back(*v);
    }

    // BFS through all non-protein voxels
    while let Some(v) = queue.pop_front() {
        let current_dist = distances[pocket.linear_idx(v[0], v[1], v[2])];

        for (dx, dy, dz) in NEIGHBORS_6 {
            let nx = v[0] as i32 + dx;
            let ny = v[1] as i32 + dy;
            let nz = v[2] as i32 + dz;

            if pocket.in_bounds(nx, ny, nz) {
                let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                // Only traverse non-protein space
                if !protein_solid.get(nx, ny, nz) {
                    let neighbor_idx = pocket.linear_idx(nx, ny, nz);
                    let new_dist = current_dist + spacing;
                    if new_dist < distances[neighbor_idx] {
                        distances[neighbor_idx] = new_dist;
                        queue.push_back([nx, ny, nz]);
                    }
                }
            }
        }
    }

    // Collect distances for pocket voxels
    let mut pocket_surface_distances = Vec::new();
    for z in 0..pocket.dims[2] {
        for y in 0..pocket.dims[1] {
            for x in 0..pocket.dims[0] {
                if pocket.get(x, y, z) {
                    let d = distances[pocket.linear_idx(x, y, z)];
                    if d < f32::MAX && d > 0.0 {
                        pocket_surface_distances.push(d);
                    }
                }
            }
        }
    }

    if pocket_surface_distances.is_empty() {
        None
    } else {
        Some(percentile_nearest_rank(&pocket_surface_distances, percentile))
    }
}

/// Compute percentile using nearest-rank method
pub fn percentile_nearest_rank(data: &[f32], p: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Nearest-rank: index = ceil(p/100 * N) - 1, clamped to [0, N-1]
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f32).ceil() as usize;
    let index = rank.saturating_sub(1).min(n - 1);

    sorted[index]
}

// =============================================================================
// SHAPE METRICS FROM POINT CLOUDS (PCA-based)
// =============================================================================

/// Result of PCA-based shape analysis
#[derive(Debug, Clone, Default)]
pub struct ShapeAnalysisResult {
    /// Aspect ratio: λ_max / λ_min (>= 1.0, 1.0 = perfect sphere)
    pub aspect_ratio: f64,
    /// Sphericity: λ_min / λ_max (0-1, 1.0 = perfect sphere)
    pub sphericity: f64,
    /// Eigenvalues [λ1, λ2, λ3] in descending order
    pub eigenvalues: [f64; 3],
    /// Principal axes (eigenvectors, columns)
    pub principal_axes: [[f64; 3]; 3],
    /// Bounding box dimensions along principal axes [d1, d2, d3]
    pub oriented_bbox_dims: [f64; 3],
    /// Volume of oriented bounding box (Å³)
    pub oriented_bbox_volume: f64,
}

/// Compute shape metrics from a point cloud using PCA
///
/// This computes REAL aspect ratio and sphericity from the actual
/// spatial distribution of points, not heuristics.
///
/// # Arguments
/// * `points` - 3D coordinates of event centers
///
/// # Returns
/// * `ShapeAnalysisResult` with eigenvalue-based metrics
pub fn compute_shape_from_points(points: &[[f32; 3]]) -> Option<ShapeAnalysisResult> {
    if points.len() < 4 {
        // Need at least 4 points for meaningful PCA
        return None;
    }

    let n = points.len() as f64;

    // Step 1: Compute centroid
    let mut centroid = [0.0f64; 3];
    for p in points {
        centroid[0] += p[0] as f64;
        centroid[1] += p[1] as f64;
        centroid[2] += p[2] as f64;
    }
    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;

    // Step 2: Build covariance matrix
    let mut cov = [[0.0f64; 3]; 3];
    for p in points {
        let dx = p[0] as f64 - centroid[0];
        let dy = p[1] as f64 - centroid[1];
        let dz = p[2] as f64 - centroid[2];

        cov[0][0] += dx * dx;
        cov[0][1] += dx * dy;
        cov[0][2] += dx * dz;
        cov[1][1] += dy * dy;
        cov[1][2] += dy * dz;
        cov[2][2] += dz * dz;
    }

    // Symmetrize
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // Normalize
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] /= n;
        }
    }

    // Step 3: Eigenvalue decomposition using Jacobi iteration
    let (eigenvalues, eigenvectors) = jacobi_eigendecomposition_3x3(cov)?;

    // Sort eigenvalues descending
    let mut indices = [0, 1, 2];
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_eigenvalues = [
        eigenvalues[indices[0]].max(1e-10),
        eigenvalues[indices[1]].max(1e-10),
        eigenvalues[indices[2]].max(1e-10),
    ];

    let sorted_eigenvectors = [
        eigenvectors[indices[0]],
        eigenvectors[indices[1]],
        eigenvectors[indices[2]],
    ];

    // Step 4: Compute shape metrics
    let lambda_max = sorted_eigenvalues[0];
    let lambda_min = sorted_eigenvalues[2];

    // Aspect ratio from eigenvalues (sqrt because eigenvalues are variances)
    let aspect_ratio = (lambda_max / lambda_min).sqrt().max(1.0);
    let sphericity = (1.0 / aspect_ratio).clamp(0.0, 1.0);

    // Step 5: Compute oriented bounding box dimensions
    // Project all points onto principal axes and find extents
    let mut min_proj = [f64::INFINITY; 3];
    let mut max_proj = [f64::NEG_INFINITY; 3];

    for p in points {
        let dx = p[0] as f64 - centroid[0];
        let dy = p[1] as f64 - centroid[1];
        let dz = p[2] as f64 - centroid[2];

        for i in 0..3 {
            let proj = dx * sorted_eigenvectors[i][0]
                     + dy * sorted_eigenvectors[i][1]
                     + dz * sorted_eigenvectors[i][2];
            min_proj[i] = min_proj[i].min(proj);
            max_proj[i] = max_proj[i].max(proj);
        }
    }

    let oriented_bbox_dims = [
        (max_proj[0] - min_proj[0]).max(0.1),
        (max_proj[1] - min_proj[1]).max(0.1),
        (max_proj[2] - min_proj[2]).max(0.1),
    ];

    let oriented_bbox_volume = oriented_bbox_dims[0] * oriented_bbox_dims[1] * oriented_bbox_dims[2];

    Some(ShapeAnalysisResult {
        aspect_ratio,
        sphericity,
        eigenvalues: sorted_eigenvalues,
        principal_axes: sorted_eigenvectors,
        oriented_bbox_dims,
        oriented_bbox_volume,
    })
}

/// Jacobi eigenvalue decomposition for 3x3 symmetric matrix
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i] is the i-th eigenvector
fn jacobi_eigendecomposition_3x3(mut a: [[f64; 3]; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // Identity

    const MAX_ITER: usize = 50;
    const EPSILON: f64 = 1e-15;

    for _ in 0..MAX_ITER {
        // Find largest off-diagonal element
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;

        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_off {
                    max_off = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < EPSILON {
            break; // Converged
        }

        // Compute rotation angle
        let diff = a[q][q] - a[p][p];
        let t = if diff.abs() < EPSILON {
            if a[p][q] >= 0.0 { 1.0 } else { -1.0 }
        } else {
            let phi = diff / (2.0 * a[p][q]);
            if phi >= 0.0 {
                1.0 / (phi + (1.0 + phi * phi).sqrt())
            } else {
                1.0 / (phi - (1.0 + phi * phi).sqrt())
            }
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let tau = s / (1.0 + c);

        // Apply rotation to A
        let h = t * a[p][q];
        a[p][p] -= h;
        a[q][q] += h;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for j in 0..p {
            let g = a[j][p];
            let h = a[j][q];
            a[j][p] = g - s * (h + g * tau);
            a[j][q] = h + s * (g - h * tau);
        }
        for j in (p + 1)..q {
            let g = a[p][j];
            let h = a[j][q];
            a[p][j] = g - s * (h + g * tau);
            a[j][q] = h + s * (g - h * tau);
        }
        for j in (q + 1)..3 {
            let g = a[p][j];
            let h = a[q][j];
            a[p][j] = g - s * (h + g * tau);
            a[q][j] = h + s * (g - h * tau);
        }

        // Symmetrize
        for i in 0..3 {
            for j in (i + 1)..3 {
                a[j][i] = a[i][j];
            }
        }

        // Apply rotation to eigenvectors
        for j in 0..3 {
            let g = v[j][p];
            let h = v[j][q];
            v[j][p] = g - s * (h + g * tau);
            v[j][q] = h + s * (g - h * tau);
        }
    }

    // Extract eigenvalues (diagonal of A)
    let eigenvalues = [a[0][0], a[1][1], a[2][2]];

    // Transpose V to get eigenvectors as rows
    let eigenvectors = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];

    Some((eigenvalues, eigenvectors))
}

// =============================================================================
// VOLUME STATISTICS FROM EVENT TRAJECTORY
// =============================================================================

/// Result of volume trajectory analysis
#[derive(Debug, Clone, Default)]
pub struct VolumeStatisticsResult {
    /// Minimum volume observed (Å³)
    pub volume_min: f64,
    /// Maximum volume observed (Å³)
    pub volume_max: f64,
    /// Mean volume (Å³)
    pub volume_mean: f64,
    /// Median volume (Å³)
    pub volume_median: f64,
    /// Standard deviation of volume (Å³)
    pub volume_std: f64,
    /// Breathing amplitude: max - min (Å³)
    pub breathing_amplitude: f64,
    /// Coefficient of variation: std / mean
    pub volume_cv: f64,
    /// Number of volume measurements
    pub n_measurements: usize,
}

/// Compute volume statistics from event volumes
///
/// This computes REAL breathing amplitude from actual volume variance,
/// not a heuristic like volume_mean * 0.6.
///
/// # Arguments
/// * `volumes` - Volume measurements from events (Å³)
pub fn compute_volume_statistics(volumes: &[f64]) -> Option<VolumeStatisticsResult> {
    if volumes.is_empty() {
        return None;
    }

    let n = volumes.len();
    let n_f = n as f64;

    // Sort for median and percentiles
    let mut sorted: Vec<f64> = volumes.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let volume_min = sorted[0];
    let volume_max = sorted[n - 1];

    // Mean
    let volume_mean = volumes.iter().sum::<f64>() / n_f;

    // Median
    let volume_median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    // Standard deviation
    let variance = volumes.iter()
        .map(|v| (v - volume_mean).powi(2))
        .sum::<f64>() / n_f;
    let volume_std = variance.sqrt();

    // Breathing amplitude (actual range, not heuristic)
    let breathing_amplitude = volume_max - volume_min;

    // Coefficient of variation
    let volume_cv = if volume_mean > 1e-10 {
        volume_std / volume_mean
    } else {
        0.0
    };

    Some(VolumeStatisticsResult {
        volume_min,
        volume_max,
        volume_mean,
        volume_median,
        volume_std,
        breathing_amplitude,
        volume_cv,
        n_measurements: n,
    })
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test PCA on a perfect sphere (should have aspect_ratio ≈ 1)
    #[test]
    fn test_shape_sphere() {
        // Generate uniformly distributed points on a sphere using Fibonacci lattice
        // This method avoids the polar clustering issue of naive spherical coordinates
        let n = 1000;
        let mut points = Vec::new();

        let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // ~2.4 radians

        for i in 0..n {
            // y goes from +1 to -1 uniformly
            let y = 1.0 - (i as f64 / (n - 1) as f64) * 2.0;
            let radius_at_y = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f64;

            let x = radius_at_y * theta.cos();
            let z = radius_at_y * theta.sin();

            // Scale to radius 10
            points.push([10.0 * x as f32, 10.0 * y as f32, 10.0 * z as f32]);
        }

        let result = compute_shape_from_points(&points).unwrap();

        // Sphere should have aspect_ratio close to 1
        assert!(result.aspect_ratio < 1.15, "Sphere aspect_ratio should be ~1, got {}", result.aspect_ratio);
        assert!(result.sphericity > 0.85, "Sphere sphericity should be ~1, got {}", result.sphericity);
    }

    /// Test PCA on an elongated rod (should have high aspect_ratio)
    #[test]
    fn test_shape_rod() {
        // Generate points along a rod: 100 points along X, ±1 in Y and Z
        let mut points = Vec::new();
        for i in 0..100 {
            let x = i as f32;
            points.push([x, 0.0, 0.0]);
            points.push([x, 1.0, 0.0]);
            points.push([x, -1.0, 0.0]);
            points.push([x, 0.0, 1.0]);
            points.push([x, 0.0, -1.0]);
        }

        let result = compute_shape_from_points(&points).unwrap();

        // Rod should have high aspect_ratio
        assert!(result.aspect_ratio > 5.0, "Rod aspect_ratio should be high, got {}", result.aspect_ratio);
        assert!(result.sphericity < 0.3, "Rod sphericity should be low, got {}", result.sphericity);
    }

    /// Test volume statistics
    #[test]
    fn test_volume_statistics() {
        let volumes = vec![100.0, 150.0, 200.0, 180.0, 120.0, 300.0, 250.0, 50.0];

        let result = compute_volume_statistics(&volumes).unwrap();

        assert_eq!(result.volume_min, 50.0);
        assert_eq!(result.volume_max, 300.0);
        assert_eq!(result.breathing_amplitude, 250.0); // Real amplitude, not fake!
        assert_eq!(result.n_measurements, 8);

        // Mean should be 168.75
        let expected_mean = (100.0 + 150.0 + 200.0 + 180.0 + 120.0 + 300.0 + 250.0 + 50.0) / 8.0;
        assert!((result.volume_mean - expected_mean).abs() < 0.001);
    }

    /// Test volume statistics with single value
    #[test]
    fn test_volume_statistics_single() {
        let volumes = vec![200.0];
        let result = compute_volume_statistics(&volumes).unwrap();

        assert_eq!(result.volume_min, 200.0);
        assert_eq!(result.volume_max, 200.0);
        assert_eq!(result.breathing_amplitude, 0.0);
        assert_eq!(result.volume_std, 0.0);
    }

    /// Test 1: mouth area on a flat opening (3x3 slab at x=1)
    #[test]
    fn test_mouth_area_flat_opening() {
        // Grid: 7x7x7, voxel size dx=1.0
        let dims = [7, 7, 7];
        let spacing = 1.0;

        // Pocket mask: 3x3x1 slab at x=1, y=2..4, z=2..4
        let mut pocket = BoolGrid::new(dims, false);
        for y in 2..=4 {
            for z in 2..=4 {
                pocket.set(1, y, z, true);
            }
        }

        // Bulk solvent: x=0 plane is solvent
        let mut bulk_solvent = BoolGrid::new(dims, false);
        for y in 0..7 {
            for z in 0..7 {
                bulk_solvent.set(0, y, z, true);
            }
        }

        let (mouth_voxels, solvent_faces) = compute_mouth_voxels_and_faces(&pocket, &bulk_solvent);

        // Each of the 9 pocket voxels has 1 face adjacent to x=0 solvent
        assert_eq!(mouth_voxels.len(), 9, "Should have 9 mouth voxels");

        let total_faces: usize = solvent_faces.iter().sum();
        assert_eq!(total_faces, 9, "Should have 9 solvent-facing faces");

        let (n_openings, areas) = compute_openings(&mouth_voxels, &solvent_faces, &pocket, spacing);
        assert_eq!(n_openings, 1, "Should have 1 opening");
        assert!((areas[0] - 9.0).abs() < 0.001, "Mouth area should be 9.0");
    }

    /// Test 2: depth on a 1-voxel-wide tunnel (length 11)
    #[test]
    fn test_depth_tunnel() {
        // Grid: 1x1x11, dx=1.0
        // Pocket: all voxels z=0..10
        // Mouth at z=0 (adjacent to bulk solvent at z=-1, but we simulate with z=0 being the opening)

        let dims = [1, 1, 11];
        let spacing = 1.0;

        let mut pocket = BoolGrid::new(dims, false);
        for z in 0..11 {
            pocket.set(0, 0, z, true);
        }

        // Mouth voxel is at z=0
        let mouth_voxels = vec![[0, 0, 0]];

        let depths = compute_depths_bfs(&pocket, &mouth_voxels, spacing);

        // Should have 11 depth values: 0, 1, 2, ..., 10
        assert_eq!(depths.len(), 11, "Should have 11 depth values");

        // p95 using nearest-rank on values 0..10
        // N=11, p=95, rank = ceil(0.95*11) = ceil(10.45) = 11, index = 11-1=10
        let p95 = percentile_nearest_rank(&depths, 95.0);
        assert!((p95 - 10.0).abs() < 0.001, "p95 should be 10.0, got {}", p95);
    }

    /// Test 3: two openings with different areas
    #[test]
    fn test_two_openings() {
        // Grid: 11x11x1, dx=1.0
        // Opening A: 2x2 at (1..2, 1..2, 0) - 4 voxels
        // Opening B: 3x3 at (5..7, 5..7, 0) - 9 voxels
        // Bulk solvent at z=-1 (simulated by having z=0 boundary be the mouth)

        let dims = [11, 11, 2];
        let spacing = 1.0;

        let mut pocket = BoolGrid::new(dims, false);

        // Opening A: 2x2
        for x in 1..=2 {
            for y in 1..=2 {
                pocket.set(x, y, 0, true);
            }
        }

        // Opening B: 3x3
        for x in 5..=7 {
            for y in 5..=7 {
                pocket.set(x, y, 0, true);
            }
        }

        // Bulk solvent at z=1
        let mut bulk_solvent = BoolGrid::new(dims, false);
        for x in 0..11 {
            for y in 0..11 {
                bulk_solvent.set(x, y, 1, true);
            }
        }

        let (mouth_voxels, solvent_faces) = compute_mouth_voxels_and_faces(&pocket, &bulk_solvent);

        // Total mouth voxels: 4 + 9 = 13
        assert_eq!(mouth_voxels.len(), 13, "Should have 13 mouth voxels");

        let (n_openings, areas) = compute_openings(&mouth_voxels, &solvent_faces, &pocket, spacing);

        assert_eq!(n_openings, 2, "Should have 2 openings");

        let total: f32 = areas.iter().sum();
        assert!((total - 13.0).abs() < 0.001, "Total area should be 13.0");

        let max_area = areas.iter().cloned().fold(0.0f32, f32::max);
        assert!((max_area - 9.0).abs() < 0.001, "Max opening area should be 9.0");
    }

    #[test]
    fn test_percentile_nearest_rank() {
        // Values 0..10 (11 values)
        let data: Vec<f32> = (0..=10).map(|x| x as f32).collect();

        // p50: rank = ceil(0.5*11) = 6, index = 5
        assert!((percentile_nearest_rank(&data, 50.0) - 5.0).abs() < 0.001);

        // p95: rank = ceil(0.95*11) = 11, index = 10
        assert!((percentile_nearest_rank(&data, 95.0) - 10.0).abs() < 0.001);

        // p90: rank = ceil(0.9*11) = 10, index = 9
        assert!((percentile_nearest_rank(&data, 90.0) - 9.0).abs() < 0.001);

        // p100: rank = ceil(1.0*11) = 11, index = 10
        assert!((percentile_nearest_rank(&data, 100.0) - 10.0).abs() < 0.001);
    }
}
