//! Cryptic site feature vector definition
//!
//! 16-dimensional feature vector capturing dynamics, structural,
//! chemical, distance, and tertiary properties of each residue.
//!
//! ## Feature Layout
//!
//! | Index | Category  | Feature               | Description                          |
//! |-------|-----------|----------------------|--------------------------------------|
//! | 0-4   | Dynamics  | burial_change, rmsf, variance, neighbor_flexibility, burial_potential |
//! | 5-7   | Structural| ss_flexibility, sidechain_flexibility, b_factor |
//! | 8-10  | Chemical  | net_charge, hydrophobicity, h_bond_potential |
//! | 11-13 | Distance  | contact_density, sasa_change, nearest_charged_dist |
//! | 14-15 | Tertiary  | interface_score, allosteric_proximity |
//!
//! ## Buffer Layout (40 dimensions)
//!
//! - [0-15]: Current feature values
//! - [16-31]: Velocity (delta from previous frame)
//! - [32-39]: Padding (zeros)

use serde::{Deserialize, Serialize};

/// Cryptic site feature vector (16 dimensions)
///
/// Captures the key physical and chemical properties that indicate
/// cryptic binding site potential for each residue.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CrypticFeatures {
    // === Dynamics Features (5) ===
    /// Change in burial upon conformational sampling
    pub burial_change: f32,
    /// Root mean square fluctuation from ensemble
    pub rmsf: f32,
    /// Variance of position across ensemble
    pub variance: f32,
    /// Average flexibility of neighboring residues
    pub neighbor_flexibility: f32,
    /// Predicted burial potential (from ANM/NOVA)
    pub burial_potential: f32,

    // === Structural Features (3) ===
    /// Secondary structure flexibility score
    pub ss_flexibility: f32,
    /// Side chain rotamer flexibility
    pub sidechain_flexibility: f32,
    /// Crystallographic B-factor (normalized)
    pub b_factor: f32,

    // === Chemical Features (3) ===
    /// Net charge of residue
    pub net_charge: f32,
    /// Hydrophobicity (Kyte-Doolittle scale)
    pub hydrophobicity: f32,
    /// Hydrogen bonding potential
    pub h_bond_potential: f32,

    // === Distance Features (3) ===
    /// Local contact density (neighbors within 8A)
    pub contact_density: f32,
    /// Change in solvent accessible surface area
    pub sasa_change: f32,
    /// Distance to nearest charged residue
    pub nearest_charged_dist: f32,

    // === Tertiary Features (2) ===
    /// Interface score for multi-chain proteins
    pub interface_score: f32,
    /// Proximity to known allosteric sites
    pub allosteric_proximity: f32,
}

impl CrypticFeatures {
    /// Number of base features (without velocity)
    pub const NUM_FEATURES: usize = 16;

    /// Total buffer size (features + velocity + padding)
    pub const BUFFER_SIZE: usize = 40;

    /// Encode features into 40-dim input buffer
    ///
    /// Layout: [16 features][16 velocities][8 padding]
    ///
    /// Velocities and padding are zeroed; use `encode_with_velocity`
    /// to include temporal information.
    pub fn encode_into(&self, buffer: &mut [f32; 40]) {
        buffer[0] = self.burial_change;
        buffer[1] = self.rmsf;
        buffer[2] = self.variance;
        buffer[3] = self.neighbor_flexibility;
        buffer[4] = self.burial_potential;
        buffer[5] = self.ss_flexibility;
        buffer[6] = self.sidechain_flexibility;
        buffer[7] = self.b_factor;
        buffer[8] = self.net_charge;
        buffer[9] = self.hydrophobicity;
        buffer[10] = self.h_bond_potential;
        buffer[11] = self.contact_density;
        buffer[12] = self.sasa_change;
        buffer[13] = self.nearest_charged_dist;
        buffer[14] = self.interface_score;
        buffer[15] = self.allosteric_proximity;

        // Velocity slots (16-31) and padding (32-39) - set to zeros
        for i in 16..40 {
            buffer[i] = 0.0;
        }
    }

    /// Encode with velocity information from previous frame
    ///
    /// Computes temporal derivatives (deltas) from the previous
    /// feature state and stores them in slots 16-31.
    pub fn encode_with_velocity(&self, prev: &CrypticFeatures, buffer: &mut [f32; 40]) {
        // First encode current features
        self.encode_into(buffer);

        // Compute velocities (deltas from previous)
        buffer[16] = self.burial_change - prev.burial_change;
        buffer[17] = self.rmsf - prev.rmsf;
        buffer[18] = self.variance - prev.variance;
        buffer[19] = self.neighbor_flexibility - prev.neighbor_flexibility;
        buffer[20] = self.burial_potential - prev.burial_potential;
        buffer[21] = self.ss_flexibility - prev.ss_flexibility;
        buffer[22] = self.sidechain_flexibility - prev.sidechain_flexibility;
        buffer[23] = self.b_factor - prev.b_factor;
        buffer[24] = self.net_charge - prev.net_charge;
        buffer[25] = self.hydrophobicity - prev.hydrophobicity;
        buffer[26] = self.h_bond_potential - prev.h_bond_potential;
        buffer[27] = self.contact_density - prev.contact_density;
        buffer[28] = self.sasa_change - prev.sasa_change;
        buffer[29] = self.nearest_charged_dist - prev.nearest_charged_dist;
        buffer[30] = self.interface_score - prev.interface_score;
        buffer[31] = self.allosteric_proximity - prev.allosteric_proximity;
    }

    /// Create from raw array (for testing and deserialization)
    pub fn from_array(arr: &[f32; 16]) -> Self {
        Self {
            burial_change: arr[0],
            rmsf: arr[1],
            variance: arr[2],
            neighbor_flexibility: arr[3],
            burial_potential: arr[4],
            ss_flexibility: arr[5],
            sidechain_flexibility: arr[6],
            b_factor: arr[7],
            net_charge: arr[8],
            hydrophobicity: arr[9],
            h_bond_potential: arr[10],
            contact_density: arr[11],
            sasa_change: arr[12],
            nearest_charged_dist: arr[13],
            interface_score: arr[14],
            allosteric_proximity: arr[15],
        }
    }

    /// Convert to raw array
    pub fn to_array(&self) -> [f32; 16] {
        [
            self.burial_change,
            self.rmsf,
            self.variance,
            self.neighbor_flexibility,
            self.burial_potential,
            self.ss_flexibility,
            self.sidechain_flexibility,
            self.b_factor,
            self.net_charge,
            self.hydrophobicity,
            self.h_bond_potential,
            self.contact_density,
            self.sasa_change,
            self.nearest_charged_dist,
            self.interface_score,
            self.allosteric_proximity,
        ]
    }

    /// Normalize features to [0, 1] range
    ///
    /// Applies sigmoid to unbounded features and linear scaling
    /// to bounded features based on expected physical ranges.
    pub fn normalize(&mut self) {
        // Apply sigmoid to unbounded features
        self.burial_change = sigmoid(self.burial_change);
        self.variance = sigmoid(self.variance);
        self.neighbor_flexibility = sigmoid(self.neighbor_flexibility);

        // Clamp bounded features to expected ranges
        self.rmsf = self.rmsf.clamp(0.0, 10.0) / 10.0;
        self.b_factor = self.b_factor.clamp(0.0, 100.0) / 100.0;
        self.contact_density = self.contact_density.clamp(0.0, 30.0) / 30.0;
        self.sasa_change = sigmoid(self.sasa_change);
        self.nearest_charged_dist = self.nearest_charged_dist.clamp(0.0, 20.0) / 20.0;

        // Normalize remaining features
        self.burial_potential = sigmoid(self.burial_potential);
        self.ss_flexibility = self.ss_flexibility.clamp(0.0, 1.0);
        self.sidechain_flexibility = self.sidechain_flexibility.clamp(0.0, 1.0);

        // Charge: map [-1, 1] to [0, 1]
        self.net_charge = (self.net_charge.clamp(-1.0, 1.0) + 1.0) / 2.0;

        // Hydrophobicity: Kyte-Doolittle scale is roughly [-4.5, 4.5]
        self.hydrophobicity = (self.hydrophobicity.clamp(-4.5, 4.5) + 4.5) / 9.0;

        self.h_bond_potential = self.h_bond_potential.clamp(0.0, 1.0);
        self.interface_score = self.interface_score.clamp(0.0, 1.0);
        self.allosteric_proximity = self.allosteric_proximity.clamp(0.0, 1.0);
    }

    /// Check if all features are within valid normalized range [0, 1]
    pub fn is_normalized(&self) -> bool {
        let arr = self.to_array();
        arr.iter().all(|&v| v >= 0.0 && v <= 1.0)
    }
}

/// Sigmoid activation function
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_roundtrip() {
        let features = CrypticFeatures {
            burial_change: 0.5,
            rmsf: 1.2,
            net_charge: -1.0,
            ..Default::default()
        };

        let mut buffer = [0.0f32; 40];
        features.encode_into(&mut buffer);

        assert!((buffer[0] - 0.5).abs() < 1e-6);
        assert!((buffer[1] - 1.2).abs() < 1e-6);
        assert!((buffer[8] - (-1.0)).abs() < 1e-6);

        // Check padding is zero
        for i in 16..40 {
            assert_eq!(buffer[i], 0.0);
        }
    }

    #[test]
    fn test_velocity_encoding() {
        let prev = CrypticFeatures {
            burial_change: 0.3,
            rmsf: 1.0,
            ..Default::default()
        };

        let curr = CrypticFeatures {
            burial_change: 0.5,
            rmsf: 1.5,
            ..Default::default()
        };

        let mut buffer = [0.0f32; 40];
        curr.encode_with_velocity(&prev, &mut buffer);

        // Check current values
        assert!((buffer[0] - 0.5).abs() < 1e-6);
        assert!((buffer[1] - 1.5).abs() < 1e-6);

        // Check velocity slots
        assert!((buffer[16] - 0.2).abs() < 1e-6); // burial_change velocity
        assert!((buffer[17] - 0.5).abs() < 1e-6); // rmsf velocity

        // Padding should still be zero
        for i in 32..40 {
            assert_eq!(buffer[i], 0.0);
        }
    }

    #[test]
    fn test_from_array_roundtrip() {
        let arr: [f32; 16] = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ];

        let features = CrypticFeatures::from_array(&arr);
        let back = features.to_array();

        for i in 0..16 {
            assert!((arr[i] - back[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize() {
        let mut features = CrypticFeatures {
            burial_change: 2.0,   // Should become ~0.88 via sigmoid
            rmsf: 15.0,           // Should clamp to 1.0
            variance: -5.0,       // Should become ~0.007 via sigmoid
            net_charge: -1.0,     // Should become 0.0
            hydrophobicity: 4.5,  // Should become 1.0
            contact_density: 50.0, // Should clamp to 1.0
            ..Default::default()
        };

        features.normalize();

        // Check sigmoid applied correctly
        assert!(features.burial_change > 0.8 && features.burial_change < 0.95);
        assert!(features.variance < 0.1);

        // Check clamping
        assert!((features.rmsf - 1.0).abs() < 1e-6);
        assert!((features.contact_density - 1.0).abs() < 1e-6);

        // Check charge normalization
        assert!((features.net_charge - 0.0).abs() < 1e-6);

        // Check hydrophobicity normalization
        assert!((features.hydrophobicity - 1.0).abs() < 1e-6);

        // All values should be in [0, 1]
        assert!(features.is_normalized());
    }

    #[test]
    fn test_default_is_zero() {
        let features = CrypticFeatures::default();
        let arr = features.to_array();

        for val in arr {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_constants() {
        assert_eq!(CrypticFeatures::NUM_FEATURES, 16);
        assert_eq!(CrypticFeatures::BUFFER_SIZE, 40);
    }
}
