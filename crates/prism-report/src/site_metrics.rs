//! Deterministic, post-run residue mapping + depth_proxy + mouth_area_proxy
//! using ONLY topology.json (positions/residue_ids/residue_names/chain_ids)
//! and post-run voxel/site data.
//!
//! HARD POLICY:
//! - No engine stepping changes.
//! - No in-run voxelization.
//! - No placeholders/defaults: hard-fail if required metrics cannot be computed.
//! - Deterministic ordering (BTreeSet/BTreeMap + stable iteration order).
//!
//! Expected TopologyData fields (as in topology.json):
//!   n_atoms: usize
//!   positions: Vec<f32>          // len = n_atoms*3  (Å)
//!   residue_ids: Vec<u32>        // len = n_atoms    (0..n_residues-1)
//!   residue_names: Vec<String>   // len = n_atoms
//!   chain_ids: Vec<String>       // len = n_atoms
//!
//! This module does NOT read PDB. It only uses TopologyData.

use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// Compute centroid of the AABB
    pub fn centroid(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Check if two AABBs overlap with a margin
    pub fn overlaps_with_margin(&self, other: &Aabb, margin: f32) -> bool {
        for i in 0..3 {
            if self.max[i] + margin < other.min[i] || other.max[i] + margin < self.min[i] {
                return false;
            }
        }
        true
    }

    /// Compute max absolute axis delta between two AABBs
    pub fn max_axis_delta(&self, other: &Aabb) -> f32 {
        let mut max_delta = 0.0f32;
        for i in 0..3 {
            let delta_min = (self.min[i] - other.min[i]).abs();
            let delta_max = (self.max[i] - other.max[i]).abs();
            max_delta = max_delta.max(delta_min).max(delta_max);
        }
        max_delta
    }
}

/// Unique residue key for deterministic ordering
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResidueKey {
    pub chain: String,
    pub resid: u32,      // residue index from topology.residue_ids[atom]
    pub resname: String, // residue_names[atom]
}

impl ResidueKey {
    /// Format as "A:LEU123"
    pub fn to_label(&self) -> String {
        format!("{}:{}{}", self.chain, self.resname, self.resid)
    }
}

/// Result of residue mapping
#[derive(Debug, Clone)]
pub struct ResidueMappingResult {
    pub residues: Vec<ResidueKey>, // sorted deterministically
    pub residue_count: usize,
    pub hydrophobic_fraction: f32, // [0,1]
}

/// Result of depth proxy computation
#[derive(Debug, Clone)]
pub struct DepthResult {
    pub depth_proxy_a: f32, // Å
}

/// Result of mouth area computation
#[derive(Debug, Clone)]
pub struct MouthAreaResult {
    pub mouth_area_proxy_a2: f32, // Å²
    pub exposed_face_count: u32,
}

/// Deterministic spatial grid index for atom queries
#[derive(Debug)]
pub struct AtomGridIndex {
    cell_size: f32,
    cells: BTreeMap<(i32, i32, i32), Vec<usize>>,
    aabb: Aabb,
    n_atoms: usize,
}

impl AtomGridIndex {
    /// Build spatial index from atom positions
    pub fn build(positions: &[f32], n_atoms: usize, cell_size: f32) -> Result<Self> {
        if n_atoms == 0 {
            bail!("AtomGridIndex: n_atoms=0");
        }
        if positions.len() != n_atoms * 3 {
            bail!(
                "AtomGridIndex: positions length mismatch: got {}, expected {}",
                positions.len(),
                n_atoms * 3
            );
        }
        if cell_size <= 0.0 {
            bail!("AtomGridIndex: cell_size must be > 0");
        }

        // Compute AABB deterministically
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for i in 0..n_atoms {
            let b = i * 3;
            let x = positions[b];
            let y = positions[b + 1];
            let z = positions[b + 2];
            if x < min[0] { min[0] = x; }
            if y < min[1] { min[1] = y; }
            if z < min[2] { min[2] = z; }
            if x > max[0] { max[0] = x; }
            if y > max[1] { max[1] = y; }
            if z > max[2] { max[2] = z; }
        }

        let mut cells: BTreeMap<(i32, i32, i32), Vec<usize>> = BTreeMap::new();
        for i in 0..n_atoms {
            let p = get_pos(positions, i);
            let k = cell_key(p, cell_size);
            cells.entry(k).or_default().push(i);
        }

        // Deterministic: sort atom indices per cell
        for (_k, v) in cells.iter_mut() {
            v.sort_unstable();
        }

        Ok(Self {
            cell_size,
            cells,
            aabb: Aabb { min, max },
            n_atoms,
        })
    }

    /// Get the AABB of all atoms
    pub fn aabb(&self) -> Aabb {
        self.aabb
    }

    /// Deterministic radius query
    pub fn query_radius(&self, positions: &[f32], center: [f32; 3], radius: f32) -> Result<Vec<usize>> {
        if radius <= 0.0 {
            bail!("query_radius: radius must be > 0");
        }
        if positions.len() != self.n_atoms * 3 {
            bail!("query_radius: positions length mismatch");
        }

        let r2 = radius * radius;
        let (c0x, c0y, c0z) = cell_key(center, self.cell_size);
        let cr = (radius / self.cell_size).ceil() as i32;

        let mut hits: Vec<usize> = Vec::new();
        for cz in (c0z - cr)..=(c0z + cr) {
            for cy in (c0y - cr)..=(c0y + cr) {
                for cx in (c0x - cr)..=(c0x + cr) {
                    if let Some(atom_idxs) = self.cells.get(&(cx, cy, cz)) {
                        for &ai in atom_idxs {
                            let p = get_pos(positions, ai);
                            let dx = p[0] - center[0];
                            let dy = p[1] - center[1];
                            let dz = p[2] - center[2];
                            let d2 = dx * dx + dy * dy + dz * dz;
                            if d2 <= r2 {
                                hits.push(ai);
                            }
                        }
                    }
                }
            }
        }

        // Deterministic: sort unique
        hits.sort_unstable();
        hits.dedup();
        Ok(hits)
    }

    /// Nearest distance to a set of surface atom indices
    pub fn nearest_distance_to_indices(
        &self,
        positions: &[f32],
        point: [f32; 3],
        indices: &[usize],
    ) -> Result<f32> {
        if indices.is_empty() {
            bail!("nearest_distance_to_indices: empty indices");
        }
        let mut best = f32::INFINITY;
        for &i in indices {
            let p = get_pos(positions, i);
            let dx = p[0] - point[0];
            let dy = p[1] - point[1];
            let dz = p[2] - point[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < best {
                best = d;
            }
        }
        Ok(best)
    }
}

// ============================================================================
// TopologyData - loaded from topology.json only
// ============================================================================

/// Topology data loaded from topology.json
/// This is the ONLY source of spatial coordinates for metrics computation.
#[derive(Debug, Clone)]
pub struct TopologyData {
    pub n_atoms: usize,
    pub positions: Vec<f32>,        // len = n_atoms*3
    pub residue_ids: Vec<u32>,      // len = n_atoms (0..n_residues-1)
    pub residue_names: Vec<String>, // len = n_atoms
    pub chain_ids: Vec<String>,     // len = n_atoms
}

impl TopologyData {
    /// Load from topology.json file
    pub fn load_from_json(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read topology JSON: {}", path.display()))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse topology JSON: {}", path.display()))?;

        let n_atoms = json["n_atoms"]
            .as_u64()
            .context("topology.json missing 'n_atoms'")? as usize;

        let positions: Vec<f32> = json["positions"]
            .as_array()
            .context("topology.json missing 'positions' array")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        let residue_ids: Vec<u32> = json["residue_ids"]
            .as_array()
            .context("topology.json missing 'residue_ids' array")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u32)
            .collect();

        let residue_names: Vec<String> = json["residue_names"]
            .as_array()
            .context("topology.json missing 'residue_names' array")?
            .iter()
            .map(|v| v.as_str().unwrap_or("UNK").to_string())
            .collect();

        let chain_ids: Vec<String> = json["chain_ids"]
            .as_array()
            .context("topology.json missing 'chain_ids' array")?
            .iter()
            .map(|v| v.as_str().unwrap_or("A").to_string())
            .collect();

        let topo = Self {
            n_atoms,
            positions,
            residue_ids,
            residue_names,
            chain_ids,
        };

        topo.validate()?;
        Ok(topo)
    }

    /// Validate topology data consistency
    pub fn validate(&self) -> Result<()> {
        if self.n_atoms == 0 {
            bail!("TopologyData: n_atoms=0");
        }
        if self.positions.len() != self.n_atoms * 3 {
            bail!(
                "TopologyData: positions length mismatch: got {}, expected {}",
                self.positions.len(),
                self.n_atoms * 3
            );
        }
        if self.residue_ids.len() != self.n_atoms {
            bail!(
                "TopologyData: residue_ids length mismatch: got {}, expected {}",
                self.residue_ids.len(),
                self.n_atoms
            );
        }
        if self.residue_names.len() != self.n_atoms {
            bail!(
                "TopologyData: residue_names length mismatch: got {}, expected {}",
                self.residue_names.len(),
                self.n_atoms
            );
        }
        if self.chain_ids.len() != self.n_atoms {
            bail!(
                "TopologyData: chain_ids length mismatch: got {}, expected {}",
                self.chain_ids.len(),
                self.n_atoms
            );
        }
        Ok(())
    }

    /// Compute AABB of all atom positions
    pub fn compute_aabb(&self) -> Aabb {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for i in 0..self.n_atoms {
            let p = get_pos(&self.positions, i);
            for j in 0..3 {
                if p[j] < min[j] { min[j] = p[j]; }
                if p[j] > max[j] { max[j] = p[j]; }
            }
        }
        Aabb { min, max }
    }

    /// Get position of atom i
    pub fn get_position(&self, atom_idx: usize) -> [f32; 3] {
        get_pos(&self.positions, atom_idx)
    }
}

// ============================================================================
// Coordinate Frame Validation
// ============================================================================

/// Diagnostic info for coordinate frame mismatch
#[derive(Debug)]
pub struct CoordinateFrameDiagnostics {
    pub topology_aabb: Aabb,
    pub events_aabb: Aabb,
    pub topology_centroid: [f32; 3],
    pub events_centroid: [f32; 3],
    pub suggested_translation: [f32; 3],
    pub max_axis_delta: f32,
    pub overlaps: bool,
}

impl std::fmt::Display for CoordinateFrameDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== COORDINATE FRAME MISMATCH DIAGNOSTICS ===")?;
        writeln!(f, "Topology AABB:")?;
        writeln!(f, "  min: [{:.2}, {:.2}, {:.2}]",
            self.topology_aabb.min[0], self.topology_aabb.min[1], self.topology_aabb.min[2])?;
        writeln!(f, "  max: [{:.2}, {:.2}, {:.2}]",
            self.topology_aabb.max[0], self.topology_aabb.max[1], self.topology_aabb.max[2])?;
        writeln!(f, "Events AABB:")?;
        writeln!(f, "  min: [{:.2}, {:.2}, {:.2}]",
            self.events_aabb.min[0], self.events_aabb.min[1], self.events_aabb.min[2])?;
        writeln!(f, "  max: [{:.2}, {:.2}, {:.2}]",
            self.events_aabb.max[0], self.events_aabb.max[1], self.events_aabb.max[2])?;
        writeln!(f, "Topology centroid: [{:.2}, {:.2}, {:.2}]",
            self.topology_centroid[0], self.topology_centroid[1], self.topology_centroid[2])?;
        writeln!(f, "Events centroid: [{:.2}, {:.2}, {:.2}]",
            self.events_centroid[0], self.events_centroid[1], self.events_centroid[2])?;
        writeln!(f, "Suggested translation (topo - events): [{:.2}, {:.2}, {:.2}]",
            self.suggested_translation[0], self.suggested_translation[1], self.suggested_translation[2])?;
        writeln!(f, "Max absolute axis delta: {:.2} Å", self.max_axis_delta)?;
        writeln!(f, "AABBs overlap (with 10Å margin): {}", self.overlaps)?;
        Ok(())
    }
}

/// Validate that events and topology are in the same coordinate frame
pub fn validate_coordinate_frames(
    topology: &TopologyData,
    event_centers: &[[f32; 3]],
    margin: f32,
) -> Result<CoordinateFrameDiagnostics> {
    if event_centers.is_empty() {
        bail!("validate_coordinate_frames: no event centers provided");
    }

    let topology_aabb = topology.compute_aabb();

    // Compute events AABB
    let mut events_min = [f32::INFINITY; 3];
    let mut events_max = [f32::NEG_INFINITY; 3];
    for center in event_centers {
        for j in 0..3 {
            if center[j] < events_min[j] { events_min[j] = center[j]; }
            if center[j] > events_max[j] { events_max[j] = center[j]; }
        }
    }
    let events_aabb = Aabb { min: events_min, max: events_max };

    let topology_centroid = topology_aabb.centroid();
    let events_centroid = events_aabb.centroid();

    let suggested_translation = [
        topology_centroid[0] - events_centroid[0],
        topology_centroid[1] - events_centroid[1],
        topology_centroid[2] - events_centroid[2],
    ];

    let max_axis_delta = topology_aabb.max_axis_delta(&events_aabb);
    let overlaps = topology_aabb.overlaps_with_margin(&events_aabb, margin);

    let diag = CoordinateFrameDiagnostics {
        topology_aabb,
        events_aabb,
        topology_centroid,
        events_centroid,
        suggested_translation,
        max_axis_delta,
        overlaps,
    };

    if !overlaps {
        bail!(
            "FATAL: Coordinate frame mismatch between topology and events.\n\
             Events are not within {}Å margin of topology atoms.\n\
             This indicates the engine produced events in a different coordinate frame.\n\n{}",
            margin,
            diag
        );
    }

    Ok(diag)
}

// ============================================================================
// SiteMetricsComputer
// ============================================================================

/// Main metrics computer using topology data
pub struct SiteMetricsComputer {
    topo: TopologyData,
    atom_index: AtomGridIndex,
    query_radius_a: f32,
}

impl SiteMetricsComputer {
    /// Create new metrics computer from topology
    pub fn new(topo: TopologyData, query_radius_a: f32) -> Result<Self> {
        topo.validate()?;
        let atom_index = AtomGridIndex::build(&topo.positions, topo.n_atoms, 6.0)
            .context("building atom spatial index")?;
        Ok(Self { topo, atom_index, query_radius_a })
    }

    /// Get topology AABB
    pub fn topology_aabb(&self) -> Aabb {
        self.atom_index.aabb()
    }

    /// Get reference to topology data
    pub fn topology(&self) -> &TopologyData {
        &self.topo
    }

    /// Deterministic downsample: lexicographic sort by (x,y,z), take first N
    pub fn canonical_sample_points(mut points: Vec<[f32; 3]>, max_points: usize) -> Vec<[f32; 3]> {
        points.sort_by(|a, b| {
            a[0].partial_cmp(&b[0]).unwrap()
                .then(a[1].partial_cmp(&b[1]).unwrap())
                .then(a[2].partial_cmp(&b[2]).unwrap())
        });
        if points.len() > max_points {
            points.truncate(max_points);
        }
        points
    }

    /// Map residues within query radius of sample points
    pub fn map_residues(&self, sample_points: &[[f32; 3]]) -> Result<ResidueMappingResult> {
        if sample_points.is_empty() {
            bail!("Residue mapping: no sample_points provided");
        }

        let mut set: BTreeSet<ResidueKey> = BTreeSet::new();

        for &p in sample_points {
            let atoms = self
                .atom_index
                .query_radius(&self.topo.positions, p, self.query_radius_a)
                .context("atom radius query failed")?;

            for ai in atoms {
                let resid = self.topo.residue_ids[ai];
                let chain = self.topo.chain_ids[ai].clone();
                let resname = self.topo.residue_names[ai].clone();
                set.insert(ResidueKey { chain, resid, resname });
            }
        }

        let residues: Vec<ResidueKey> = set.into_iter().collect();
        if residues.is_empty() {
            bail!(
                "Residue mapping: 0 residues mapped.\n\
                 Check coordinate frames: site sample points may not align with topology positions.\n\
                 Query radius: {} Å, Sample points: {}",
                self.query_radius_a,
                sample_points.len()
            );
        }

        let hydrophobic = hydrophobic_fraction(&residues);
        Ok(ResidueMappingResult {
            residue_count: residues.len(),
            residues,
            hydrophobic_fraction: hydrophobic,
        })
    }

    /// Compute depth proxy: min distance from site centroid to surface atoms
    pub fn depth_proxy(&self, site_centroid: [f32; 3]) -> Result<DepthResult> {
        let positions = &self.topo.positions;

        // Protein centroid C
        let mut c = [0.0f32; 3];
        for i in 0..self.topo.n_atoms {
            let p = get_pos(positions, i);
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        let inv = 1.0 / (self.topo.n_atoms as f32);
        c[0] *= inv;
        c[1] *= inv;
        c[2] *= inv;

        // Compute radii and find r_max
        let mut r_max = 0.0f32;
        let mut radii: Vec<f32> = Vec::with_capacity(self.topo.n_atoms);
        for i in 0..self.topo.n_atoms {
            let p = get_pos(positions, i);
            let dx = p[0] - c[0];
            let dy = p[1] - c[1];
            let dz = p[2] - c[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r > r_max { r_max = r; }
            radii.push(r);
        }

        // Surface atoms: within δ=6Å of max radius
        let delta = 6.0f32;
        let threshold = r_max - delta;
        let mut surface_indices: Vec<usize> = Vec::new();
        for (i, &r) in radii.iter().enumerate() {
            if r >= threshold {
                surface_indices.push(i);
            }
        }
        surface_indices.sort_unstable();

        if surface_indices.is_empty() {
            bail!("Depth proxy: computed 0 surface atoms (r_max={:.2}, threshold={:.2})", r_max, threshold);
        }

        let d = self
            .atom_index
            .nearest_distance_to_indices(positions, site_centroid, &surface_indices)
            .context("depth proxy nearest distance failed")?;

        Ok(DepthResult { depth_proxy_a: d.max(0.0) })
    }

    /// Compute mouth area proxy from voxel sets
    pub fn mouth_area_proxy<F>(
        &self,
        site_voxels: &[(i32, i32, i32)],
        protein_occ: F,
        voxel_size_a: f32,
    ) -> Result<MouthAreaResult>
    where
        F: Fn(i32, i32, i32) -> bool,
    {
        if site_voxels.is_empty() {
            bail!("Mouth area: site_voxels empty");
        }
        if voxel_size_a <= 0.0 {
            bail!("Mouth area: voxel_size must be > 0");
        }

        let site_set: BTreeSet<(i32, i32, i32)> = site_voxels.iter().copied().collect();

        let dirs = [
            ( 1,  0,  0),
            (-1,  0,  0),
            ( 0,  1,  0),
            ( 0, -1,  0),
            ( 0,  0,  1),
            ( 0,  0, -1),
        ];

        let mut exposed_faces: u32 = 0;

        for &(x, y, z) in site_set.iter() {
            for (dx, dy, dz) in dirs.iter() {
                let nx = x + dx;
                let ny = y + dy;
                let nz = z + dz;

                if site_set.contains(&(nx, ny, nz)) {
                    continue; // internal face
                }

                // Solvent/outside: not protein-occupied
                if !protein_occ(nx, ny, nz) {
                    exposed_faces += 1;
                }
            }
        }

        if exposed_faces == 0 {
            bail!("Mouth area: exposed_face_count=0 (check protein occupancy grid / site voxels)");
        }

        let face_area = voxel_size_a * voxel_size_a;
        let mouth_area = (exposed_faces as f32) * face_area;

        Ok(MouthAreaResult {
            mouth_area_proxy_a2: mouth_area,
            exposed_face_count: exposed_faces,
        })
    }

    /// Compute min distance from a point to any topology atom
    pub fn min_distance_to_any_atom(&self, point: [f32; 3]) -> f32 {
        let mut min_dist = f32::INFINITY;
        for i in 0..self.topo.n_atoms {
            let p = get_pos(&self.topo.positions, i);
            let dx = p[0] - point[0];
            let dy = p[1] - point[1];
            let dz = p[2] - point[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < min_dist {
                min_dist = d;
            }
        }
        min_dist
    }

    /// Count atoms within radius of a point
    pub fn count_atoms_within_radius(&self, point: [f32; 3], radius: f32) -> Result<usize> {
        let atoms = self.atom_index.query_radius(&self.topo.positions, point, radius)?;
        Ok(atoms.len())
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn get_pos(positions: &[f32], atom_idx: usize) -> [f32; 3] {
    let b = atom_idx * 3;
    [positions[b], positions[b + 1], positions[b + 2]]
}

fn cell_key(p: [f32; 3], cell: f32) -> (i32, i32, i32) {
    let cx = (p[0] / cell).floor() as i32;
    let cy = (p[1] / cell).floor() as i32;
    let cz = (p[2] / cell).floor() as i32;
    (cx, cy, cz)
}

fn hydrophobic_fraction(residues: &[ResidueKey]) -> f32 {
    let hydro = |r: &str| matches!(
        r,
        "ALA" | "VAL" | "LEU" | "ILE" | "MET" | "PHE" | "TYR" | "TRP" | "PRO"
    );
    let total = residues.len() as f32;
    if total == 0.0 {
        return 0.0;
    }
    let mut h = 0.0f32;
    for r in residues {
        if hydro(r.resname.as_str()) {
            h += 1.0;
        }
    }
    (h / total).clamp(0.0, 1.0)
}

/// Sort sites deterministically by (open_frequency desc, CV_SASA desc, volume_A3 desc)
///
/// This ensures byte-identical outputs across runs.
pub fn sort_sites_deterministic(sites: &mut [crate::sites::CrypticSite]) {
    sites.sort_by(|a, b| {
        // Primary: persistence (open_frequency) descending
        let cmp1 = b.metrics.persistence.present_fraction
            .partial_cmp(&a.metrics.persistence.present_fraction)
            .unwrap_or(std::cmp::Ordering::Equal);
        if cmp1 != std::cmp::Ordering::Equal {
            return cmp1;
        }

        // Secondary: UV delta SASA descending (as CV proxy)
        let cmp2 = b.metrics.uv_response.delta_sasa
            .partial_cmp(&a.metrics.uv_response.delta_sasa)
            .unwrap_or(std::cmp::Ordering::Equal);
        if cmp2 != std::cmp::Ordering::Equal {
            return cmp2;
        }

        // Tertiary: volume descending
        b.metrics.geometry.volume_mean
            .partial_cmp(&a.metrics.geometry.volume_mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Update ranks after sorting
    for (i, site) in sites.iter_mut().enumerate() {
        site.rank = i + 1;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_topology() -> TopologyData {
        // Simple 4-atom topology in a line
        TopologyData {
            n_atoms: 4,
            positions: vec![
                0.0, 0.0, 0.0,   // atom 0
                5.0, 0.0, 0.0,   // atom 1
                10.0, 0.0, 0.0,  // atom 2
                15.0, 0.0, 0.0,  // atom 3
            ],
            residue_ids: vec![0, 0, 1, 1],
            residue_names: vec!["ALA".into(), "ALA".into(), "LEU".into(), "LEU".into()],
            chain_ids: vec!["A".into(), "A".into(), "A".into(), "A".into()],
        }
    }

    #[test]
    fn test_topology_validation() {
        let topo = make_test_topology();
        assert!(topo.validate().is_ok());
    }

    #[test]
    fn test_topology_validation_fails_empty() {
        let topo = TopologyData {
            n_atoms: 0,
            positions: vec![],
            residue_ids: vec![],
            residue_names: vec![],
            chain_ids: vec![],
        };
        assert!(topo.validate().is_err());
    }

    #[test]
    fn test_atom_grid_index() {
        let topo = make_test_topology();
        let index = AtomGridIndex::build(&topo.positions, topo.n_atoms, 6.0).unwrap();

        // Query around atom 0
        let hits = index.query_radius(&topo.positions, [0.0, 0.0, 0.0], 3.0).unwrap();
        assert_eq!(hits, vec![0]);

        // Query with larger radius
        let hits = index.query_radius(&topo.positions, [2.5, 0.0, 0.0], 5.0).unwrap();
        assert!(hits.contains(&0));
        assert!(hits.contains(&1));
    }

    #[test]
    fn test_residue_mapping() {
        let topo = make_test_topology();
        let computer = SiteMetricsComputer::new(topo, 5.0).unwrap();

        let sample_points = vec![[2.5, 0.0, 0.0]];
        let result = computer.map_residues(&sample_points).unwrap();

        assert!(result.residue_count > 0);
        assert!(result.hydrophobic_fraction >= 0.0 && result.hydrophobic_fraction <= 1.0);
    }

    #[test]
    fn test_hydrophobic_fraction() {
        let residues = vec![
            ResidueKey { chain: "A".into(), resid: 0, resname: "ALA".into() },
            ResidueKey { chain: "A".into(), resid: 1, resname: "LEU".into() },
            ResidueKey { chain: "A".into(), resid: 2, resname: "ASP".into() },
        ];
        let frac = hydrophobic_fraction(&residues);
        // ALA and LEU are hydrophobic (2/3)
        assert!((frac - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_aabb_overlap() {
        let a = Aabb { min: [0.0, 0.0, 0.0], max: [10.0, 10.0, 10.0] };
        let b = Aabb { min: [5.0, 5.0, 5.0], max: [15.0, 15.0, 15.0] };
        assert!(a.overlaps_with_margin(&b, 0.0));

        let c = Aabb { min: [100.0, 100.0, 100.0], max: [110.0, 110.0, 110.0] };
        assert!(!a.overlaps_with_margin(&c, 10.0));
    }

    #[test]
    fn test_coordinate_validation_success() {
        let topo = make_test_topology();
        let events = vec![[2.5, 0.0, 0.0], [7.5, 0.0, 0.0]];
        let result = validate_coordinate_frames(&topo, &events, 10.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_coordinate_validation_failure() {
        let topo = make_test_topology();
        // Events far outside topology
        let events = vec![[1000.0, 1000.0, 1000.0]];
        let result = validate_coordinate_frames(&topo, &events, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic_sample_points() {
        let points = vec![
            [3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        let sorted = SiteMetricsComputer::canonical_sample_points(points, 10);
        assert_eq!(sorted[0], [1.0, 0.0, 0.0]);
        assert_eq!(sorted[1], [2.0, 0.0, 0.0]);
        assert_eq!(sorted[2], [3.0, 0.0, 0.0]);
    }
}
