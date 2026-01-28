//! Molecular dynamics simulation components

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Represents a molecule in the simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub id: usize,
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub mass: f32,
    pub charge: f32,
    pub molecule_type: MoleculeType,
}

/// Types of molecules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MoleculeType {
    Water,
    Protein,
    Ion,
    Lipid,
    Custom(String),
}

/// Molecular dynamics system
#[derive(Debug, Clone)]
pub struct MolecularSystem {
    pub molecules: Vec<Molecule>,
    pub box_size: Vector3<f32>,
    pub time: f32,
}

impl MolecularSystem {
    pub fn new(box_size: Vector3<f32>) -> Self {
        Self {
            molecules: Vec::new(),
            box_size,
            time: 0.0,
        }
    }

    /// TODO(GPU-MEC-02): GPU-accelerated force calculation
    pub fn calculate_forces(&self) -> Vec<Vector3<f32>> {
        vec![Vector3::zeros(); self.molecules.len()]
    }

    /// TODO(GPU-MEC-03): GPU-accelerated integration step
    pub fn integrate(&mut self, time_step: f32) {
        self.time += time_step;
    }
}
