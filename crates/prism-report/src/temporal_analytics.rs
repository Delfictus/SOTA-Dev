//! Temporal Analytics Module
//!
//! Computes time-dependent metrics from event streams and trajectory snapshots.
//! Integrated with the fused NHS pipeline for synchronized analysis.
//!
//! # Metrics Computed
//!
//! ## From Events Only (no trajectory needed):
//! - Spike lifetime statistics (autocorrelation)
//! - Phase-resolved opening probabilities
//! - Inter-site temporal correlation
//!
//! ## From Trajectory Snapshots:
//! - Contact breakage detection (gating residues)
//! - Rotamer state transitions (side-chain flips)
//! - Hydrogen bond inventory changes
//! - Opening mechanism identification

use crate::event_cloud::{PocketEvent, TempPhase};
use crate::sites::{
    ContactBreakage, HydrogenBondChanges, InterSiteCorrelation, PhaseResolvedStats,
    RotamerTransition, SpikeLifetimeStats, TemporalMetrics,
};
use std::collections::{HashMap, HashSet};

// =============================================================================
// TRAJECTORY FRAME (simplified for analytics)
// =============================================================================

/// Trajectory frame with atomic coordinates
#[derive(Debug, Clone)]
pub struct TrajectorySnapshot {
    /// Frame index
    pub frame_idx: usize,
    /// Timestep
    pub timestep: i32,
    /// Temperature (K)
    pub temperature: f32,
    /// Atomic positions (flat x,y,z array)
    pub positions: Vec<f32>,
    /// Number of atoms
    pub n_atoms: usize,
}

impl TrajectorySnapshot {
    /// Get position of atom i as [x, y, z]
    pub fn get_position(&self, i: usize) -> [f32; 3] {
        let base = i * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    /// Compute distance between two atoms
    pub fn distance(&self, i: usize, j: usize) -> f32 {
        let pi = self.get_position(i);
        let pj = self.get_position(j);
        let dx = pi[0] - pj[0];
        let dy = pi[1] - pj[1];
        let dz = pi[2] - pj[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// =============================================================================
// TOPOLOGY INFO (for residue/atom mapping)
// =============================================================================

/// Minimal topology info needed for temporal analytics
#[derive(Debug, Clone)]
pub struct TemporalTopology {
    /// Residue ID for each atom
    pub residue_ids: Vec<usize>,
    /// Residue names for each atom
    pub residue_names: Vec<String>,
    /// Atom names for each atom
    pub atom_names: Vec<String>,
    /// Chain IDs for each atom
    pub chain_ids: Vec<String>,
    /// CA atom indices (one per residue)
    pub ca_indices: Vec<usize>,
    /// Number of residues
    pub n_residues: usize,
}

impl TemporalTopology {
    /// Get CA index for a residue
    pub fn ca_for_residue(&self, res_id: usize) -> Option<usize> {
        self.ca_indices.iter().position(|&idx| {
            self.residue_ids.get(idx).copied() == Some(res_id)
        }).map(|i| self.ca_indices[i])
    }

    /// Get all atom indices for a residue
    pub fn atoms_for_residue(&self, res_id: usize) -> Vec<usize> {
        self.residue_ids
            .iter()
            .enumerate()
            .filter(|(_, &rid)| rid == res_id)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get sidechain atoms for chi1 calculation (N, CA, CB, XG)
    pub fn chi1_atoms(&self, res_id: usize) -> Option<[usize; 4]> {
        let atoms = self.atoms_for_residue(res_id);
        let mut n_idx = None;
        let mut ca_idx = None;
        let mut cb_idx = None;
        let mut xg_idx = None; // CG, OG, SG depending on residue

        for &ai in &atoms {
            let name = self.atom_names[ai].trim();
            match name {
                "N" => n_idx = Some(ai),
                "CA" => ca_idx = Some(ai),
                "CB" => cb_idx = Some(ai),
                "CG" | "CG1" | "OG" | "OG1" | "SG" => xg_idx = Some(ai),
                _ => {}
            }
        }

        match (n_idx, ca_idx, cb_idx, xg_idx) {
            (Some(n), Some(ca), Some(cb), Some(xg)) => Some([n, ca, cb, xg]),
            _ => None,
        }
    }

    /// Get sidechain atoms for chi2 calculation (CA, CB, CG, XD)
    pub fn chi2_atoms(&self, res_id: usize) -> Option<[usize; 4]> {
        let atoms = self.atoms_for_residue(res_id);
        let mut ca_idx = None;
        let mut cb_idx = None;
        let mut cg_idx = None;
        let mut xd_idx = None; // CD, CD1, OD1, ND1, SD depending on residue

        for &ai in &atoms {
            let name = self.atom_names[ai].trim();
            match name {
                "CA" => ca_idx = Some(ai),
                "CB" => cb_idx = Some(ai),
                "CG" | "CG1" => cg_idx = Some(ai),
                "CD" | "CD1" | "OD1" | "ND1" | "SD" => xd_idx = Some(ai),
                _ => {}
            }
        }

        match (ca_idx, cb_idx, cg_idx, xd_idx) {
            (Some(ca), Some(cb), Some(cg), Some(xd)) => Some([ca, cb, cg, xd]),
            _ => None,
        }
    }
}

// =============================================================================
// SPIKE LIFETIME ANALYSIS (Events Only)
// =============================================================================

/// Compute spike lifetime statistics from events for a site
pub fn compute_spike_lifetime_stats(
    events: &[&PocketEvent],
    site_residues: &HashSet<usize>,
    total_frames: usize,
) -> SpikeLifetimeStats {
    if events.is_empty() || total_frames == 0 {
        return SpikeLifetimeStats::default();
    }

    // Filter events that overlap with this site's residues
    let site_events: Vec<_> = events
        .iter()
        .filter(|e| {
            e.residues.iter().any(|&r| site_residues.contains(&(r as usize)))
        })
        .collect();

    if site_events.is_empty() {
        return SpikeLifetimeStats::default();
    }

    // Get unique frames where site is open
    let mut open_frames: Vec<usize> = site_events.iter().map(|e| e.frame_idx).collect();
    open_frames.sort();
    open_frames.dedup();

    // Identify contiguous opening events (runs of consecutive frames)
    let mut lifetimes: Vec<usize> = Vec::new();
    let mut current_run_start = open_frames[0];
    let mut current_run_length = 1;

    for i in 1..open_frames.len() {
        if open_frames[i] == open_frames[i - 1] + 1 {
            // Consecutive frame
            current_run_length += 1;
        } else {
            // Gap - end of run
            lifetimes.push(current_run_length);
            current_run_start = open_frames[i];
            current_run_length = 1;
        }
    }
    lifetimes.push(current_run_length); // Don't forget last run

    let n_opening_events = lifetimes.len();
    let max_lifetime = *lifetimes.iter().max().unwrap_or(&0);
    let mean_lifetime = lifetimes.iter().sum::<usize>() as f64 / n_opening_events as f64;

    // Median
    let mut sorted_lifetimes = lifetimes.clone();
    sorted_lifetimes.sort();
    let median_lifetime = if sorted_lifetimes.len() % 2 == 0 {
        let mid = sorted_lifetimes.len() / 2;
        (sorted_lifetimes[mid - 1] + sorted_lifetimes[mid]) as f64 / 2.0
    } else {
        sorted_lifetimes[sorted_lifetimes.len() / 2] as f64
    };

    // Recurrence analysis (if site opens, closes, then opens again)
    let recurrence_fraction = if n_opening_events > 1 {
        (n_opening_events - 1) as f64 / n_opening_events as f64
    } else {
        0.0
    };

    // Mean interval between recurrent openings
    let mean_recurrence_interval = if n_opening_events > 1 {
        // Compute gaps between consecutive opening events
        let mut gaps: Vec<usize> = Vec::new();
        let mut prev_end = open_frames[0];
        let mut in_run = true;

        for i in 1..open_frames.len() {
            if open_frames[i] == open_frames[i - 1] + 1 {
                // Still in run
                in_run = true;
            } else {
                // Gap found
                if in_run {
                    gaps.push(open_frames[i] - prev_end);
                }
                in_run = false;
            }
            prev_end = open_frames[i];
        }

        if gaps.is_empty() {
            None
        } else {
            Some(gaps.iter().sum::<usize>() as f64 / gaps.len() as f64)
        }
    } else {
        None
    };

    SpikeLifetimeStats {
        mean_lifetime_frames: mean_lifetime,
        median_lifetime_frames: median_lifetime,
        max_lifetime_frames: max_lifetime,
        n_opening_events,
        recurrence_fraction,
        mean_recurrence_interval,
    }
}

// =============================================================================
// PHASE-RESOLVED STATISTICS (Events Only)
// =============================================================================

/// Compute phase-resolved opening statistics
pub fn compute_phase_stats(
    events: &[&PocketEvent],
    site_residues: &HashSet<usize>,
    cold_frames: usize,
    ramp_frames: usize,
    warm_frames: usize,
) -> PhaseResolvedStats {
    if events.is_empty() {
        return PhaseResolvedStats::default();
    }

    // Filter events for this site
    let site_events: Vec<_> = events
        .iter()
        .filter(|e| {
            e.residues.iter().any(|&r| site_residues.contains(&(r as usize)))
        })
        .collect();

    if site_events.is_empty() {
        return PhaseResolvedStats::default();
    }

    // Count events by phase
    let mut cold_events: HashSet<usize> = HashSet::new();
    let mut ramp_events: HashSet<usize> = HashSet::new();
    let mut warm_events: HashSet<usize> = HashSet::new();

    let mut first_frame: Option<usize> = None;
    let mut first_temp: Option<f32> = None;
    let mut first_phase: Option<String> = None;

    for event in &site_events {
        let frame = event.frame_idx;

        // Track first opening
        if first_frame.is_none() || frame < first_frame.unwrap() {
            first_frame = Some(frame);
            first_phase = Some(format!("{}", event.temp_phase));
            // Estimate temperature from phase (rough approximation)
            first_temp = Some(match event.temp_phase {
                TempPhase::Cold => 50.0,
                TempPhase::Ramp => 175.0,
                TempPhase::Warm => 300.0,
            });
        }

        match event.temp_phase {
            TempPhase::Cold => { cold_events.insert(frame); }
            TempPhase::Ramp => { ramp_events.insert(frame); }
            TempPhase::Warm => { warm_events.insert(frame); }
        }
    }

    // Compute probabilities (fraction of frames in phase where site is open)
    let cold_prob = if cold_frames > 0 {
        cold_events.len() as f64 / cold_frames as f64
    } else {
        0.0
    };

    let ramp_prob = if ramp_frames > 0 {
        ramp_events.len() as f64 / ramp_frames as f64
    } else {
        0.0
    };

    let warm_prob = if warm_frames > 0 {
        warm_events.len() as f64 / warm_frames as f64
    } else {
        0.0
    };

    // Dominant phase
    let dominant_phase = if cold_prob >= ramp_prob && cold_prob >= warm_prob {
        Some("cold".to_string())
    } else if ramp_prob >= cold_prob && ramp_prob >= warm_prob {
        Some("ramp".to_string())
    } else {
        Some("warm".to_string())
    };

    PhaseResolvedStats {
        first_opening_frame: first_frame,
        first_opening_temp_k: first_temp,
        first_opening_phase: first_phase,
        cold_phase_probability: cold_prob,
        ramp_phase_probability: ramp_prob,
        warm_phase_probability: warm_prob,
        dominant_phase,
    }
}

// =============================================================================
// CONTACT BREAKAGE DETECTION (Requires Trajectory)
// =============================================================================

/// Contact distance threshold (Å) - CA-CA distance for contact
const CONTACT_THRESHOLD: f32 = 8.0;
/// Contact breakage threshold - distance increase to count as broken
const BREAKAGE_THRESHOLD: f32 = 4.0;

/// Detect contact breakages between two snapshots
pub fn detect_contact_breakages(
    before: &TrajectorySnapshot,
    after: &TrajectorySnapshot,
    topology: &TemporalTopology,
    site_residues: &HashSet<usize>,
    frame_idx: usize,
) -> Vec<ContactBreakage> {
    let mut breakages = Vec::new();

    // Get all residue pairs where at least one is in the site
    let residue_ids: HashSet<usize> = topology.residue_ids.iter().copied().collect();

    for &res_i in &residue_ids {
        for &res_j in &residue_ids {
            if res_j <= res_i {
                continue; // Avoid duplicates
            }

            // At least one residue should be in or near the site
            let i_in_site = site_residues.contains(&res_i);
            let j_in_site = site_residues.contains(&res_j);

            if !i_in_site && !j_in_site {
                continue;
            }

            // Get CA indices
            let ca_i = match topology.ca_for_residue(res_i) {
                Some(idx) => idx,
                None => continue,
            };
            let ca_j = match topology.ca_for_residue(res_j) {
                Some(idx) => idx,
                None => continue,
            };

            // Compute distances
            let dist_before = before.distance(ca_i, ca_j);
            let dist_after = after.distance(ca_i, ca_j);

            // Check for contact breakage
            if dist_before < CONTACT_THRESHOLD && dist_after - dist_before > BREAKAGE_THRESHOLD {
                // Find residue names
                let res_i_name = topology.residue_names.get(ca_i)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| "UNK".to_string());
                let res_j_name = topology.residue_names.get(ca_j)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| "UNK".to_string());

                // Is this a gating contact? (one in site, one outside)
                let is_gating = i_in_site != j_in_site;

                breakages.push(ContactBreakage {
                    res_i,
                    res_j,
                    res_i_name,
                    res_j_name,
                    distance_before_a: dist_before,
                    distance_after_a: dist_after,
                    breakage_frame: frame_idx,
                    is_gating_contact: is_gating,
                });
            }
        }
    }

    // Sort by distance change (most significant first)
    breakages.sort_by(|a, b| {
        let delta_a = a.distance_after_a - a.distance_before_a;
        let delta_b = b.distance_after_a - b.distance_before_a;
        delta_b.partial_cmp(&delta_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    breakages
}

// =============================================================================
// ROTAMER TRACKING (Requires Trajectory)
// =============================================================================

/// Compute dihedral angle from four atom positions (degrees)
fn compute_dihedral(p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], p4: [f32; 3]) -> f32 {
    // Vectors
    let b1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let b2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
    let b3 = [p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]];

    // Cross products
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);

    // Normalize
    let n1_len = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
    let n2_len = (n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]).sqrt();

    if n1_len < 1e-6 || n2_len < 1e-6 {
        return 0.0;
    }

    let n1 = [n1[0] / n1_len, n1[1] / n1_len, n1[2] / n1_len];
    let n2 = [n2[0] / n2_len, n2[1] / n2_len, n2[2] / n2_len];

    // Angle
    let cos_angle = dot(n1, n2).clamp(-1.0, 1.0);
    let b2_len = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
    let m1 = cross(n1, [b2[0] / b2_len, b2[1] / b2_len, b2[2] / b2_len]);
    let sin_angle = dot(m1, n2);

    sin_angle.atan2(cos_angle).to_degrees()
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rotamer transition threshold (degrees)
const ROTAMER_THRESHOLD: f32 = 60.0;

/// Detect rotamer transitions between two snapshots
pub fn detect_rotamer_transitions(
    before: &TrajectorySnapshot,
    after: &TrajectorySnapshot,
    topology: &TemporalTopology,
    site_residues: &HashSet<usize>,
    frame_idx: usize,
) -> Vec<RotamerTransition> {
    let mut transitions = Vec::new();

    // Focus on aromatic and bulky residues that can occlude pockets
    let occluding_residues = ["TYR", "PHE", "TRP", "LEU", "ILE", "MET", "HIS"];

    for &res_id in site_residues {
        // Get chi1 atoms
        let chi1_atoms = match topology.chi1_atoms(res_id) {
            Some(atoms) => atoms,
            None => continue,
        };

        // Get residue name
        let res_name = topology.residue_names.get(chi1_atoms[1])
            .map(|s| s.clone())
            .unwrap_or_else(|| "UNK".to_string());

        // Compute chi1 before and after
        let chi1_before = compute_dihedral(
            before.get_position(chi1_atoms[0]),
            before.get_position(chi1_atoms[1]),
            before.get_position(chi1_atoms[2]),
            before.get_position(chi1_atoms[3]),
        );

        let chi1_after = compute_dihedral(
            after.get_position(chi1_atoms[0]),
            after.get_position(chi1_atoms[1]),
            after.get_position(chi1_atoms[2]),
            after.get_position(chi1_atoms[3]),
        );

        // Compute delta (handle wraparound)
        let mut delta_chi1 = chi1_after - chi1_before;
        if delta_chi1 > 180.0 {
            delta_chi1 -= 360.0;
        } else if delta_chi1 < -180.0 {
            delta_chi1 += 360.0;
        }

        // Check for significant transition
        if delta_chi1.abs() > ROTAMER_THRESHOLD {
            // Optional chi2
            let (chi2_before, chi2_after) = if let Some(chi2_atoms) = topology.chi2_atoms(res_id) {
                let c2b = compute_dihedral(
                    before.get_position(chi2_atoms[0]),
                    before.get_position(chi2_atoms[1]),
                    before.get_position(chi2_atoms[2]),
                    before.get_position(chi2_atoms[3]),
                );
                let c2a = compute_dihedral(
                    after.get_position(chi2_atoms[0]),
                    after.get_position(chi2_atoms[1]),
                    after.get_position(chi2_atoms[2]),
                    after.get_position(chi2_atoms[3]),
                );
                (Some(c2b), Some(c2a))
            } else {
                (None, None)
            };

            let is_occluding = occluding_residues.iter()
                .any(|&r| res_name.to_uppercase().contains(r));

            transitions.push(RotamerTransition {
                residue_id: res_id,
                residue_name: res_name,
                chi1_before_deg: chi1_before,
                chi1_after_deg: chi1_after,
                chi2_before_deg: chi2_before,
                chi2_after_deg: chi2_after,
                transition_frame: frame_idx,
                delta_chi1_deg: delta_chi1,
                is_occluding,
            });
        }
    }

    // Sort by delta magnitude
    transitions.sort_by(|a, b| {
        b.delta_chi1_deg.abs().partial_cmp(&a.delta_chi1_deg.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    transitions
}

// =============================================================================
// HYDROGEN BOND ANALYSIS (Requires Trajectory)
// =============================================================================

/// H-bond distance threshold (Å)
const HBOND_DIST_THRESHOLD: f32 = 3.5;

/// Simple H-bond count based on donor-acceptor distances
/// (Full H-bond detection would require hydrogen positions and angles)
pub fn count_approximate_hbonds(
    snapshot: &TrajectorySnapshot,
    topology: &TemporalTopology,
    site_residues: &HashSet<usize>,
) -> (usize, usize) {
    // Donor atoms: N, NZ, NE, NH1, NH2, ND1, ND2, NE1, NE2, OG, OH
    // Acceptor atoms: O, OD1, OD2, OE1, OE2, OG, OH, ND1, NE2
    let donors = ["N", "NZ", "NE", "NH1", "NH2", "ND1", "ND2", "NE1", "NE2", "OG", "OH"];
    let acceptors = ["O", "OD1", "OD2", "OE1", "OE2", "OG", "OH", "ND1", "NE2"];

    let mut protein_protein = 0;
    let mut protein_water = 0; // Can't compute without water - placeholder

    // Get site atoms
    let site_atoms: HashSet<usize> = site_residues
        .iter()
        .flat_map(|&res_id| topology.atoms_for_residue(res_id))
        .collect();

    // Find potential H-bonds between site atoms
    for &ai in &site_atoms {
        let name_i = topology.atom_names.get(ai).map(|s| s.trim()).unwrap_or("");
        let is_donor_i = donors.iter().any(|&d| name_i == d);
        let is_acceptor_i = acceptors.iter().any(|&a| name_i == a);

        if !is_donor_i && !is_acceptor_i {
            continue;
        }

        for &aj in &site_atoms {
            if aj <= ai {
                continue;
            }

            let name_j = topology.atom_names.get(aj).map(|s| s.trim()).unwrap_or("");
            let is_donor_j = donors.iter().any(|&d| name_j == d);
            let is_acceptor_j = acceptors.iter().any(|&a| name_j == a);

            // Need donor-acceptor pair
            let valid_pair = (is_donor_i && is_acceptor_j) || (is_acceptor_i && is_donor_j);
            if !valid_pair {
                continue;
            }

            let dist = snapshot.distance(ai, aj);
            if dist < HBOND_DIST_THRESHOLD {
                protein_protein += 1;
            }
        }
    }

    (protein_protein, protein_water)
}

/// Compute hydrogen bond changes between snapshots
pub fn compute_hbond_changes(
    before: &TrajectorySnapshot,
    after: &TrajectorySnapshot,
    topology: &TemporalTopology,
    site_residues: &HashSet<usize>,
    frame_idx: usize,
) -> HydrogenBondChanges {
    let (pp_before, pw_before) = count_approximate_hbonds(before, topology, site_residues);
    let (pp_after, pw_after) = count_approximate_hbonds(after, topology, site_residues);

    HydrogenBondChanges {
        protein_water_before: pw_before,
        protein_water_after: pw_after,
        protein_water_delta: pw_after as i32 - pw_before as i32,
        intra_protein_before: pp_before,
        intra_protein_after: pp_after,
        intra_protein_delta: pp_after as i32 - pp_before as i32,
        measurement_frame: frame_idx,
    }
}

// =============================================================================
// INTER-SITE CORRELATION (Events Only)
// =============================================================================

/// Compute temporal correlation between two sites
pub fn compute_inter_site_correlation(
    events: &[&PocketEvent],
    site_a_residues: &HashSet<usize>,
    site_b_residues: &HashSet<usize>,
    site_a_id: &str,
    site_b_id: &str,
    total_frames: usize,
) -> InterSiteCorrelation {
    // Build binary opening vectors
    let mut a_open = vec![false; total_frames];
    let mut b_open = vec![false; total_frames];

    for event in events {
        let in_a = event.residues.iter().any(|&r| site_a_residues.contains(&(r as usize)));
        let in_b = event.residues.iter().any(|&r| site_b_residues.contains(&(r as usize)));

        if event.frame_idx < total_frames {
            if in_a {
                a_open[event.frame_idx] = true;
            }
            if in_b {
                b_open[event.frame_idx] = true;
            }
        }
    }

    // Compute Pearson correlation
    let n = total_frames as f64;
    let mean_a = a_open.iter().filter(|&&x| x).count() as f64 / n;
    let mean_b = b_open.iter().filter(|&&x| x).count() as f64 / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..total_frames {
        let xa = if a_open[i] { 1.0 } else { 0.0 };
        let xb = if b_open[i] { 1.0 } else { 0.0 };
        cov += (xa - mean_a) * (xb - mean_b);
        var_a += (xa - mean_a) * (xa - mean_a);
        var_b += (xb - mean_b) * (xb - mean_b);
    }

    let correlation = if var_a > 0.0 && var_b > 0.0 {
        cov / (var_a.sqrt() * var_b.sqrt())
    } else {
        0.0
    };

    // Compute mean lag (when A opens relative to B)
    let mut lags: Vec<i64> = Vec::new();
    for i in 0..total_frames {
        if a_open[i] {
            // Find nearest B opening
            let mut nearest_b: Option<i64> = None;
            for j in 0..total_frames {
                if b_open[j] {
                    let lag = j as i64 - i as i64;
                    if nearest_b.is_none() || lag.abs() < nearest_b.unwrap().abs() {
                        nearest_b = Some(lag);
                    }
                }
            }
            if let Some(lag) = nearest_b {
                lags.push(lag);
            }
        }
    }

    let mean_lag = if lags.is_empty() {
        0.0
    } else {
        lags.iter().sum::<i64>() as f64 / lags.len() as f64
    };

    let is_synchronized = correlation.abs() > 0.5;

    let interpretation = if correlation > 0.5 {
        format!("Sites open together (positive coupling, r={:.2})", correlation)
    } else if correlation < -0.5 {
        format!("Sites anti-correlate (one opens when other closes, r={:.2})", correlation)
    } else if mean_lag.abs() > 10.0 {
        format!("Sequential opening (lag={:.0} frames)", mean_lag)
    } else {
        format!("Independent opening (r={:.2})", correlation)
    };

    InterSiteCorrelation {
        site_a: site_a_id.to_string(),
        site_b: site_b_id.to_string(),
        correlation,
        mean_lag_frames: mean_lag,
        is_synchronized,
        interpretation,
    }
}

// =============================================================================
// MECHANISM SUMMARY GENERATION
// =============================================================================

/// Generate a human-readable mechanism summary
pub fn generate_mechanism_summary(
    phase_stats: &PhaseResolvedStats,
    contact_breakages: &[ContactBreakage],
    rotamer_transitions: &[RotamerTransition],
    hbond_changes: Option<&HydrogenBondChanges>,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Opening timing
    if let (Some(frame), Some(phase)) = (phase_stats.first_opening_frame, &phase_stats.first_opening_phase) {
        parts.push(format!("Site first opened at frame {} ({} phase)", frame, phase));
    }

    // Gating contacts
    let gating_contacts: Vec<_> = contact_breakages.iter().filter(|c| c.is_gating_contact).collect();
    if !gating_contacts.is_empty() {
        let c = &gating_contacts[0];
        parts.push(format!(
            "Gating contact {}{}-{}{} broke ({:.1}Å → {:.1}Å)",
            c.res_i_name, c.res_i, c.res_j_name, c.res_j,
            c.distance_before_a, c.distance_after_a
        ));
    }

    // Rotamer flips
    let occluding_rotamers: Vec<_> = rotamer_transitions.iter().filter(|r| r.is_occluding).collect();
    if !occluding_rotamers.is_empty() {
        let r = &occluding_rotamers[0];
        parts.push(format!(
            "{}{} χ1 flip ({:.0}° → {:.0}°, Δ={:.0}°)",
            r.residue_name, r.residue_id,
            r.chi1_before_deg, r.chi1_after_deg, r.delta_chi1_deg
        ));
    }

    // H-bond changes
    if let Some(hb) = hbond_changes {
        if hb.intra_protein_delta != 0 {
            parts.push(format!("{} intra-protein H-bonds {}",
                hb.intra_protein_delta.abs(),
                if hb.intra_protein_delta < 0 { "lost" } else { "formed" }
            ));
        }
    }

    // Dominant phase
    if let Some(ref phase) = phase_stats.dominant_phase {
        parts.push(format!("Most open during {} phase", phase));
    }

    if parts.is_empty() {
        "Opening mechanism not determined (insufficient trajectory data)".to_string()
    } else {
        parts.join("; ")
    }
}

// =============================================================================
// FULL TEMPORAL METRICS COMPUTATION
// =============================================================================

/// Compute complete temporal metrics for a site
pub fn compute_temporal_metrics(
    events: &[&PocketEvent],
    site_residues: &HashSet<usize>,
    topology: Option<&TemporalTopology>,
    snapshots: Option<&[TrajectorySnapshot]>,
    cold_frames: usize,
    ramp_frames: usize,
    warm_frames: usize,
) -> TemporalMetrics {
    let total_frames = cold_frames + ramp_frames + warm_frames;

    // Event-based metrics (always computed)
    let lifetime = compute_spike_lifetime_stats(events, site_residues, total_frames);
    let phase_stats = compute_phase_stats(events, site_residues, cold_frames, ramp_frames, warm_frames);

    // Trajectory-based metrics (only if snapshots provided)
    let (contact_breakages, rotamer_transitions, hbond_changes) =
        if let (Some(topo), Some(snaps)) = (topology, snapshots) {
            if snaps.len() >= 2 {
                // Find snapshot pair around first opening
                let first_frame = phase_stats.first_opening_frame.unwrap_or(0);

                // Find before/after snapshots
                let before_idx = snaps.iter()
                    .rposition(|s| s.frame_idx < first_frame)
                    .unwrap_or(0);
                let after_idx = snaps.iter()
                    .position(|s| s.frame_idx >= first_frame)
                    .unwrap_or(snaps.len() - 1);

                let before = &snaps[before_idx];
                let after = &snaps[after_idx.min(snaps.len() - 1)];

                let contacts = detect_contact_breakages(before, after, topo, site_residues, first_frame);
                let rotamers = detect_rotamer_transitions(before, after, topo, site_residues, first_frame);
                let hbonds = Some(compute_hbond_changes(before, after, topo, site_residues, first_frame));

                (contacts, rotamers, hbonds)
            } else {
                (Vec::new(), Vec::new(), None)
            }
        } else {
            (Vec::new(), Vec::new(), None)
        };

    // Generate mechanism summary
    let mechanism_summary = Some(generate_mechanism_summary(
        &phase_stats,
        &contact_breakages,
        &rotamer_transitions,
        hbond_changes.as_ref(),
    ));

    TemporalMetrics {
        lifetime,
        phase_stats,
        contact_breakages,
        rotamer_transitions,
        hbond_changes,
        mechanism_summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_cloud::AblationPhase;

    fn make_test_event(frame: usize, residues: Vec<u32>, temp_phase: TempPhase) -> PocketEvent {
        PocketEvent {
            center_xyz: [0.0, 0.0, 0.0],
            volume_a3: 100.0,
            spike_count: 1,
            phase: AblationPhase::CryoUv,
            temp_phase,
            replicate_id: 0,
            frame_idx: frame,
            residues,
            confidence: 0.8,
            wavelength_nm: None,
        }
    }

    #[test]
    fn test_spike_lifetime_single_run() {
        let events = vec![
            make_test_event(10, vec![1, 2], TempPhase::Ramp),
            make_test_event(11, vec![1, 2], TempPhase::Ramp),
            make_test_event(12, vec![1, 2], TempPhase::Ramp),
        ];
        let refs: Vec<&PocketEvent> = events.iter().collect();
        let site_residues: HashSet<usize> = [1, 2].iter().copied().collect();

        let stats = compute_spike_lifetime_stats(&refs, &site_residues, 100);

        assert_eq!(stats.n_opening_events, 1);
        assert_eq!(stats.max_lifetime_frames, 3);
        assert!((stats.mean_lifetime_frames - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_spike_lifetime_multiple_runs() {
        let events = vec![
            // First opening: frames 10-12
            make_test_event(10, vec![1, 2], TempPhase::Ramp),
            make_test_event(11, vec![1, 2], TempPhase::Ramp),
            make_test_event(12, vec![1, 2], TempPhase::Ramp),
            // Gap: frames 13-19
            // Second opening: frames 20-21
            make_test_event(20, vec![1, 2], TempPhase::Warm),
            make_test_event(21, vec![1, 2], TempPhase::Warm),
        ];
        let refs: Vec<&PocketEvent> = events.iter().collect();
        let site_residues: HashSet<usize> = [1, 2].iter().copied().collect();

        let stats = compute_spike_lifetime_stats(&refs, &site_residues, 100);

        assert_eq!(stats.n_opening_events, 2);
        assert_eq!(stats.max_lifetime_frames, 3);
        assert!((stats.mean_lifetime_frames - 2.5).abs() < 0.01); // (3 + 2) / 2
        assert!(stats.recurrence_fraction > 0.0);
    }

    #[test]
    fn test_phase_stats() {
        let events = vec![
            make_test_event(5, vec![1], TempPhase::Cold),
            make_test_event(15, vec![1], TempPhase::Ramp),
            make_test_event(16, vec![1], TempPhase::Ramp),
            make_test_event(25, vec![1], TempPhase::Warm),
        ];
        let refs: Vec<&PocketEvent> = events.iter().collect();
        let site_residues: HashSet<usize> = [1].iter().copied().collect();

        let stats = compute_phase_stats(&refs, &site_residues, 10, 10, 10);

        assert_eq!(stats.first_opening_frame, Some(5));
        assert_eq!(stats.first_opening_phase, Some("cold".to_string()));
        assert!((stats.cold_phase_probability - 0.1).abs() < 0.01); // 1/10
        assert!((stats.ramp_phase_probability - 0.2).abs() < 0.01); // 2/10
        assert!((stats.warm_phase_probability - 0.1).abs() < 0.01); // 1/10
    }

    #[test]
    fn test_dihedral_calculation() {
        // Test with a simple 90 degree dihedral
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [1.0, 1.0, 0.0];
        let p4 = [1.0, 1.0, 1.0];

        let angle = compute_dihedral(p1, p2, p3, p4);
        // Should be approximately 90 degrees
        assert!((angle.abs() - 90.0).abs() < 1.0);
    }
}
