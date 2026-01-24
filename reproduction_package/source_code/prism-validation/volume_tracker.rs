//! Volume Tracking for Cryptic Binding Site Detection
//!
//! Tracks pocket volume and SASA over trajectory frames to identify cryptic sites.
//! Cryptic sites are characterized by high volume variance (they "breathe").
//!
//! # Algorithm
//!
//! 1. For each frame, detect pockets using alpha-sphere or grid-based methods
//! 2. Track volume, SASA, and centroid for each pocket
//! 3. Match pockets across frames using centroid proximity
//! 4. Identify cryptic sites by high coefficient of variation (CV > 20%)
//!
//! # Key Metrics
//!
//! - **Volume CV**: coefficient of variation = std/mean
//!   - CV > 20% suggests cryptic (dynamic) pocket
//!   - CV < 10% suggests constitutive pocket
//!
//! - **Open Frequency**: fraction of frames where pocket is detectable
//!   - 5-90% is characteristic of cryptic sites
//!   - >90% suggests always-open pocket (not cryptic)
//!
//! - **Breathing Amplitude**: max_volume - min_volume
//!   - Larger amplitude = more dramatic opening

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Single frame of volume data for a pocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeFrame {
    /// Frame index (0-based)
    pub frame: usize,
    /// Time in picoseconds (if known)
    pub time_ps: Option<f64>,
    /// Pocket volume in Å³
    pub volume: f64,
    /// Solvent accessible surface area in Å²
    pub sasa: f64,
    /// Pocket centroid [x, y, z] in Å
    pub centroid: [f64; 3],
    /// Number of residues defining the pocket
    pub n_residues: usize,
    /// Whether pocket is considered "open" in this frame
    pub is_open: bool,
    /// Druggability score for this frame (if computed)
    pub druggability: Option<f64>,
}

/// Time series of volume measurements for a single pocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeTimeSeries {
    /// Unique pocket identifier
    pub pocket_id: String,
    /// Residues that define this pocket (consensus across frames)
    pub defining_residues: Vec<i32>,
    /// Per-frame volume data
    pub frames: Vec<VolumeFrame>,
    /// Statistical summary
    pub stats: VolumeStatistics,
    /// Is this a cryptic site?
    pub is_cryptic: bool,
    /// Cryptic classification confidence
    pub cryptic_confidence: f64,
}

/// Statistical summary of volume time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeStatistics {
    /// Number of frames analyzed
    pub n_frames: usize,
    /// Number of frames where pocket was detected (volume > threshold)
    pub n_open_frames: usize,
    /// Open frequency = n_open_frames / n_frames
    pub open_frequency: f64,
    /// Mean volume (Å³) when open
    pub mean_volume: f64,
    /// Standard deviation of volume
    pub std_volume: f64,
    /// Coefficient of variation = std/mean
    pub cv_volume: f64,
    /// Minimum volume observed
    pub min_volume: f64,
    /// Maximum volume observed
    pub max_volume: f64,
    /// Breathing amplitude = max - min
    pub breathing_amplitude: f64,
    /// Mean SASA (Å²) when open
    pub mean_sasa: f64,
    /// Mean druggability when open
    pub mean_druggability: Option<f64>,
    /// Frame with maximum volume (representative open state)
    pub max_volume_frame: usize,
    /// Frame with minimum volume (representative closed state)
    pub min_volume_frame: usize,
}

impl VolumeTimeSeries {
    /// Create from a sequence of volume frames
    pub fn from_frames(pocket_id: String, defining_residues: Vec<i32>, frames: Vec<VolumeFrame>) -> Self {
        let stats = Self::compute_statistics(&frames);

        // Classify as cryptic based on volume CV and open frequency
        // Cryptic: CV > 20% AND open frequency 5-90%
        let is_cryptic = stats.cv_volume > 0.20
            && stats.open_frequency >= 0.05
            && stats.open_frequency <= 0.90;

        // Confidence based on how clearly it meets criteria
        let cv_confidence = if stats.cv_volume > 0.30 { 1.0 }
            else if stats.cv_volume > 0.20 { 0.8 }
            else if stats.cv_volume > 0.15 { 0.5 }
            else { 0.2 };

        let freq_confidence = if stats.open_frequency >= 0.10 && stats.open_frequency <= 0.80 {
            1.0
        } else if stats.open_frequency >= 0.05 && stats.open_frequency <= 0.90 {
            0.7
        } else {
            0.3
        };

        let cryptic_confidence = (cv_confidence + freq_confidence) / 2.0;

        Self {
            pocket_id,
            defining_residues,
            frames,
            stats,
            is_cryptic,
            cryptic_confidence,
        }
    }

    /// Compute statistics from frames
    fn compute_statistics(frames: &[VolumeFrame]) -> VolumeStatistics {
        if frames.is_empty() {
            return VolumeStatistics {
                n_frames: 0,
                n_open_frames: 0,
                open_frequency: 0.0,
                mean_volume: 0.0,
                std_volume: 0.0,
                cv_volume: 0.0,
                min_volume: 0.0,
                max_volume: 0.0,
                breathing_amplitude: 0.0,
                mean_sasa: 0.0,
                mean_druggability: None,
                max_volume_frame: 0,
                min_volume_frame: 0,
            };
        }

        let n_frames = frames.len();
        let open_frames: Vec<&VolumeFrame> = frames.iter().filter(|f| f.is_open).collect();
        let n_open_frames = open_frames.len();
        let open_frequency = n_open_frames as f64 / n_frames as f64;

        // Volume statistics (only for open frames)
        let (mean_volume, std_volume, min_volume, max_volume, min_frame, max_frame) =
            if open_frames.is_empty() {
                (0.0, 0.0, 0.0, 0.0, 0, 0)
            } else {
                let volumes: Vec<f64> = open_frames.iter().map(|f| f.volume).collect();
                let mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
                let variance = if volumes.len() > 1 {
                    volumes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (volumes.len() - 1) as f64
                } else {
                    0.0
                };
                let std = variance.sqrt();

                let min = volumes.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = volumes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                let min_idx = open_frames.iter().position(|f| (f.volume - min).abs() < 0.01).unwrap_or(0);
                let max_idx = open_frames.iter().position(|f| (f.volume - max).abs() < 0.01).unwrap_or(0);

                (mean, std, min, max, open_frames[min_idx].frame, open_frames[max_idx].frame)
            };

        let cv_volume = if mean_volume > 0.0 { std_volume / mean_volume } else { 0.0 };
        let breathing_amplitude = max_volume - min_volume;

        // SASA statistics
        let mean_sasa = if !open_frames.is_empty() {
            open_frames.iter().map(|f| f.sasa).sum::<f64>() / open_frames.len() as f64
        } else {
            0.0
        };

        // Druggability statistics
        let drug_scores: Vec<f64> = open_frames.iter()
            .filter_map(|f| f.druggability)
            .collect();
        let mean_druggability = if !drug_scores.is_empty() {
            Some(drug_scores.iter().sum::<f64>() / drug_scores.len() as f64)
        } else {
            None
        };

        VolumeStatistics {
            n_frames,
            n_open_frames,
            open_frequency,
            mean_volume,
            std_volume,
            cv_volume,
            min_volume,
            max_volume,
            breathing_amplitude,
            mean_sasa,
            mean_druggability,
            max_volume_frame: max_frame,
            min_volume_frame: min_frame,
        }
    }

    /// Update statistics after modifying frames
    pub fn recompute_statistics(&mut self) {
        self.stats = Self::compute_statistics(&self.frames);
    }

    /// Get the representative "open" structure frame
    pub fn get_open_representative_frame(&self) -> Option<&VolumeFrame> {
        self.frames.get(self.stats.max_volume_frame)
    }

    /// Get the representative "closed" structure frame
    pub fn get_closed_representative_frame(&self) -> Option<&VolumeFrame> {
        self.frames.get(self.stats.min_volume_frame)
    }

    /// Get top N frames by volume (open states)
    pub fn get_top_open_frames(&self, n: usize) -> Vec<&VolumeFrame> {
        let mut open_frames: Vec<&VolumeFrame> = self.frames.iter()
            .filter(|f| f.is_open)
            .collect();

        open_frames.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap_or(std::cmp::Ordering::Equal));
        open_frames.into_iter().take(n).collect()
    }
}

/// Tracks pockets across trajectory frames
pub struct VolumeTracker {
    /// Minimum volume to consider as "open" (Å³)
    pub min_open_volume: f64,
    /// Maximum distance between centroids to consider same pocket (Å)
    pub centroid_match_distance: f64,
    /// Minimum frames a pocket must appear in to be tracked
    pub min_frames_for_tracking: usize,
    /// All tracked pockets
    pockets: HashMap<String, VolumeTimeSeries>,
    /// Next pocket ID
    next_pocket_id: usize,
}

impl Default for VolumeTracker {
    fn default() -> Self {
        Self {
            min_open_volume: 100.0,  // Å³
            centroid_match_distance: 8.0,  // Å
            min_frames_for_tracking: 5,
            pockets: HashMap::new(),
            next_pocket_id: 0,
        }
    }
}

impl VolumeTracker {
    /// Create a new volume tracker with custom parameters
    pub fn new(min_open_volume: f64, centroid_match_distance: f64) -> Self {
        Self {
            min_open_volume,
            centroid_match_distance,
            min_frames_for_tracking: 5,
            pockets: HashMap::new(),
            next_pocket_id: 0,
        }
    }

    /// Add a detected pocket from a single frame
    ///
    /// The tracker will match it to existing pockets by centroid proximity.
    pub fn add_pocket_observation(
        &mut self,
        frame_idx: usize,
        time_ps: Option<f64>,
        centroid: [f64; 3],
        volume: f64,
        sasa: f64,
        residues: &[i32],
        druggability: Option<f64>,
    ) {
        let is_open = volume >= self.min_open_volume;

        let frame = VolumeFrame {
            frame: frame_idx,
            time_ps,
            volume,
            sasa,
            centroid,
            n_residues: residues.len(),
            is_open,
            druggability,
        };

        // Try to match to existing pocket
        let pocket_id = self.find_matching_pocket(&centroid)
            .unwrap_or_else(|| {
                let id = format!("pocket_{}", self.next_pocket_id);
                self.next_pocket_id += 1;
                id
            });

        // Add frame to pocket
        if let Some(pocket) = self.pockets.get_mut(&pocket_id) {
            pocket.frames.push(frame);
            // Update defining residues (merge)
            for &res in residues {
                if !pocket.defining_residues.contains(&res) {
                    pocket.defining_residues.push(res);
                }
            }
        } else {
            let pocket = VolumeTimeSeries::from_frames(
                pocket_id.clone(),
                residues.to_vec(),
                vec![frame],
            );
            self.pockets.insert(pocket_id, pocket);
        }
    }

    /// Find existing pocket that matches the given centroid
    fn find_matching_pocket(&self, centroid: &[f64; 3]) -> Option<String> {
        let dist_sq_threshold = self.centroid_match_distance * self.centroid_match_distance;

        for (id, pocket) in &self.pockets {
            // Use centroid of most recent frame
            if let Some(last_frame) = pocket.frames.last() {
                let dx = centroid[0] - last_frame.centroid[0];
                let dy = centroid[1] - last_frame.centroid[1];
                let dz = centroid[2] - last_frame.centroid[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < dist_sq_threshold {
                    return Some(id.clone());
                }
            }
        }

        None
    }

    /// Finalize tracking and compute statistics
    pub fn finalize(&mut self) {
        // Recompute statistics for all pockets
        for pocket in self.pockets.values_mut() {
            pocket.recompute_statistics();

            // Re-evaluate cryptic classification
            let stats = &pocket.stats;
            pocket.is_cryptic = stats.cv_volume > 0.20
                && stats.open_frequency >= 0.05
                && stats.open_frequency <= 0.90;
        }
    }

    /// Get all tracked pockets (including non-cryptic)
    pub fn get_all_pockets(&self) -> Vec<&VolumeTimeSeries> {
        self.pockets.values()
            .filter(|p| p.frames.len() >= self.min_frames_for_tracking)
            .collect()
    }

    /// Get only cryptic pockets, sorted by confidence
    pub fn get_cryptic_pockets(&self) -> Vec<&VolumeTimeSeries> {
        let mut cryptic: Vec<&VolumeTimeSeries> = self.pockets.values()
            .filter(|p| p.is_cryptic && p.frames.len() >= self.min_frames_for_tracking)
            .collect();

        cryptic.sort_by(|a, b| b.cryptic_confidence.partial_cmp(&a.cryptic_confidence)
            .unwrap_or(std::cmp::Ordering::Equal));

        cryptic
    }

    /// Export volume time series to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("pocket_id,frame,time_ps,volume_a3,sasa_a2,is_open,druggability,centroid_x,centroid_y,centroid_z\n");

        for pocket in self.pockets.values() {
            for frame in &pocket.frames {
                csv.push_str(&format!(
                    "{},{},{},{:.2},{:.2},{},{},{:.3},{:.3},{:.3}\n",
                    pocket.pocket_id,
                    frame.frame,
                    frame.time_ps.map(|t| t.to_string()).unwrap_or_else(|| "NA".to_string()),
                    frame.volume,
                    frame.sasa,
                    if frame.is_open { "true" } else { "false" },
                    frame.druggability.map(|d| format!("{:.3}", d)).unwrap_or_else(|| "NA".to_string()),
                    frame.centroid[0],
                    frame.centroid[1],
                    frame.centroid[2],
                ));
            }
        }

        csv
    }

    /// Export pocket summary to CSV
    pub fn summary_to_csv(&self) -> String {
        let mut csv = String::from(
            "pocket_id,n_frames,n_open,open_frequency,mean_volume,std_volume,cv_volume,\
             min_volume,max_volume,breathing_amplitude,is_cryptic,cryptic_confidence,\
             mean_druggability,residues\n"
        );

        for pocket in self.pockets.values() {
            if pocket.frames.len() < self.min_frames_for_tracking {
                continue;
            }

            let stats = &pocket.stats;
            let residues_str = pocket.defining_residues.iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(";");

            csv.push_str(&format!(
                "{},{},{},{:.3},{:.2},{:.2},{:.3},{:.2},{:.2},{:.2},{},{:.3},{},{}\n",
                pocket.pocket_id,
                stats.n_frames,
                stats.n_open_frames,
                stats.open_frequency,
                stats.mean_volume,
                stats.std_volume,
                stats.cv_volume,
                stats.min_volume,
                stats.max_volume,
                stats.breathing_amplitude,
                if pocket.is_cryptic { "true" } else { "false" },
                pocket.cryptic_confidence,
                stats.mean_druggability.map(|d| format!("{:.3}", d)).unwrap_or_else(|| "NA".to_string()),
                residues_str,
            ));
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(frame: usize, volume: f64, is_open: bool) -> VolumeFrame {
        VolumeFrame {
            frame,
            time_ps: Some(frame as f64 * 10.0),
            volume,
            sasa: volume * 0.5, // Simplified
            centroid: [0.0, 0.0, 0.0],
            n_residues: 10,
            is_open,
            druggability: Some(0.6),
        }
    }

    #[test]
    fn test_volume_statistics() {
        let frames = vec![
            make_frame(0, 200.0, true),
            make_frame(1, 250.0, true),
            make_frame(2, 180.0, true),
            make_frame(3, 300.0, true),
            make_frame(4, 50.0, false), // Closed
        ];

        let series = VolumeTimeSeries::from_frames(
            "test".to_string(),
            vec![1, 2, 3],
            frames,
        );

        assert_eq!(series.stats.n_frames, 5);
        assert_eq!(series.stats.n_open_frames, 4);
        assert!((series.stats.open_frequency - 0.8).abs() < 0.01);
        assert!((series.stats.min_volume - 180.0).abs() < 0.01);
        assert!((series.stats.max_volume - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_cryptic_classification() {
        // High variance pocket that opens and closes - should be cryptic
        // Need: CV > 0.20 AND open_frequency in 0.05..0.90
        let high_variance_frames: Vec<VolumeFrame> = (0..20).map(|i| {
            // Mix of open (volume >= 100) and closed (volume < 100) frames
            // ~60% open, ~40% closed (12 open, 8 closed)
            // Use more variable volumes to ensure CV > 0.20
            let volume = if i % 5 < 3 {
                if i % 2 == 0 { 400.0 } else { 150.0 } // High variance in open state
            } else {
                50.0 // Closed state
            };
            make_frame(i, volume, volume >= 100.0)
        }).collect();

        let high_var_series = VolumeTimeSeries::from_frames(
            "high_var".to_string(),
            vec![1, 2, 3],
            high_variance_frames,
        );

        // Verify we have the right open frequency (should be around 0.6)
        assert!(high_var_series.stats.open_frequency > 0.05, "Open freq: {}", high_var_series.stats.open_frequency);
        assert!(high_var_series.stats.open_frequency < 0.90, "Open freq: {}", high_var_series.stats.open_frequency);
        assert!(high_var_series.stats.cv_volume > 0.20, "CV: {}", high_var_series.stats.cv_volume);
        assert!(high_var_series.is_cryptic, "Should be cryptic: CV={}, open_freq={}",
            high_var_series.stats.cv_volume, high_var_series.stats.open_frequency);

        // Low variance pocket that's always open - should not be cryptic
        let low_variance_frames: Vec<VolumeFrame> = (0..20).map(|i| {
            let volume = 200.0 + (i as f64 * 2.0); // Low variance, always open
            make_frame(i, volume, true)
        }).collect();

        let low_var_series = VolumeTimeSeries::from_frames(
            "low_var".to_string(),
            vec![1, 2, 3],
            low_variance_frames,
        );

        // Should not be cryptic because open_frequency = 1.0 (> 0.90)
        assert!(!low_var_series.is_cryptic, "Should NOT be cryptic: open_freq={}",
            low_var_series.stats.open_frequency);
    }

    #[test]
    fn test_volume_tracker() {
        let mut tracker = VolumeTracker::default();

        // Add observations for a pocket that opens and closes with varying volumes
        // Use volumes above the min_open_volume threshold (100.0) that vary
        for i in 0..10 {
            let volume = 150.0 + (i as f64 * 30.0); // Volumes: 150, 180, 210, 240, ...
            tracker.add_pocket_observation(
                i,
                Some(i as f64 * 10.0),
                [10.0, 10.0, 10.0],
                volume,
                volume * 0.5,
                &[1, 2, 3, 4, 5],
                Some(0.65),
            );
        }

        tracker.finalize();

        let pockets = tracker.get_all_pockets();
        assert_eq!(pockets.len(), 1);

        // The pocket should be tracked and have 10 frames
        let pocket = &pockets[0];
        assert_eq!(pocket.frames.len(), 10);
    }

    #[test]
    fn test_csv_export() {
        let mut tracker = VolumeTracker::default();

        for i in 0..5 {
            tracker.add_pocket_observation(
                i,
                Some(i as f64 * 10.0),
                [10.0, 10.0, 10.0],
                200.0 + i as f64 * 10.0,
                100.0,
                &[1, 2, 3],
                Some(0.7),
            );
        }

        tracker.finalize();

        let csv = tracker.to_csv();
        assert!(csv.contains("pocket_id,frame"));
        assert!(csv.contains("pocket_0"));

        let summary = tracker.summary_to_csv();
        assert!(summary.contains("pocket_id,n_frames"));
        assert!(summary.contains("pocket_0"));
    }
}
