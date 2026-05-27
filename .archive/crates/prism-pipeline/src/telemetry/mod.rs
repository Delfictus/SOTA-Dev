//! Telemetry and metrics collection.

pub mod prometheus;

use chrono::Utc;
use prism_core::{CmaState, GeometryTelemetry, PhaseOutcome};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Geometry telemetry for JSON serialization.
///
/// Flattened representation of GeometryTelemetry for JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryMetrics {
    /// Geometric stress scalar (0.0 = no stress, 1.0 = critical)
    pub stress: f32,

    /// Overlap density (fraction of edges with same-color endpoints)
    pub overlap: f32,

    /// Number of geometric hotspots (high-conflict vertices)
    pub hotspots: usize,
}

impl From<&GeometryTelemetry> for GeometryMetrics {
    fn from(geom: &GeometryTelemetry) -> Self {
        Self {
            stress: geom.stress_scalar,
            overlap: geom.overlap_density,
            hotspots: geom.hotspot_count(),
        }
    }
}

/// CMA-ES telemetry for JSON serialization.
///
/// Captures evolutionary optimization metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaTelemetry {
    /// Best fitness value found
    pub best_fitness: f32,

    /// Mean fitness of current population
    pub mean_fitness: f32,

    /// Standard deviation of fitness
    pub fitness_std: f32,

    /// Step size (sigma) parameter
    pub sigma: f32,

    /// Current generation number
    pub generation: usize,

    /// Condition number of covariance matrix
    pub condition_number: f32,

    /// Convergence metric (0.0-1.0)
    pub convergence_metric: f32,
}

impl From<&CmaState> for CmaTelemetry {
    fn from(state: &CmaState) -> Self {
        Self {
            best_fitness: state.best_fitness,
            mean_fitness: state.mean_fitness,
            fitness_std: state.fitness_std,
            sigma: state.sigma,
            generation: state.generation,
            condition_number: state.covariance_condition,
            convergence_metric: state.convergence_metric,
        }
    }
}

/// Ontology phase telemetry for JSON serialization.
///
/// Captures semantic grounding metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyTelemetry {
    /// Semantic conflicts detected
    pub conflicts: u32,

    /// Semantic coherence score (0.0-1.0)
    pub coherence: f64,

    /// Concept count
    pub concept_count: usize,

    /// Ontology depth
    pub ontology_depth: usize,
}

/// MEC (Molecular Emergent Computing) telemetry for JSON serialization.
///
/// Captures molecular dynamics simulation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MecTelemetry {
    /// Free energy (Gibbs free energy)
    pub free_energy: f64,

    /// System entropy
    pub entropy: f64,

    /// Temperature (Kelvin)
    pub temperature: f64,

    /// Total energy
    pub total_energy: f64,

    /// Kinetic energy
    pub kinetic_energy: f64,

    /// Potential energy
    pub potential_energy: f64,

    /// Reaction yield
    pub reaction_yield: f64,
}

/// Biomolecular adapter telemetry for JSON serialization.
///
/// Captures protein structure prediction metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomolecularTelemetry {
    /// Prediction confidence (0.0-1.0)
    pub confidence: f64,

    /// Root mean square deviation (Angstroms)
    pub rmsd: f64,

    /// Binding affinity (kcal/mol)
    pub binding_affinity: f64,
}

/// Materials adapter telemetry for JSON serialization.
///
/// Captures materials property prediction metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialsTelemetry {
    /// Band gap (eV)
    pub band_gap: f64,

    /// Formation energy (eV/atom)
    pub formation_energy: f64,

    /// Stability score (0.0-1.0)
    pub stability: f64,
}

/// GNN (Graph Neural Network) telemetry for JSON serialization.
///
/// Captures graph embedding metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnTelemetry {
    /// Embedding dimension
    pub embedding_dim: usize,

    /// Training loss
    pub loss: f64,

    /// Prediction accuracy (0.0-1.0)
    pub accuracy: f64,
}

/// Molecular dynamics telemetry for JSON serialization.
///
/// Captures MD simulation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdTelemetry {
    /// Number of timesteps simulated
    pub timesteps: usize,

    /// Energy drift (conservation metric)
    pub energy_drift: f64,

    /// Average temperature (Kelvin)
    pub temperature_avg: f64,
}

/// Telemetry event emitted by a phase.
///
/// Implements PRISM GPU Plan §5: Telemetry Schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// ISO 8601 timestamp
    pub timestamp: String,

    /// Phase name (e.g., "Phase0-DendriticReservoir")
    pub phase: String,

    /// Phase-specific metrics
    pub metrics: HashMap<String, f64>,

    /// Outcome status
    pub outcome: String,

    /// RL action taken (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rl_action: Option<String>,

    /// RL reward received
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rl_reward: Option<f32>,

    /// Geometry telemetry from metaphysical coupling (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry: Option<GeometryMetrics>,

    /// CMA-ES optimization telemetry (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cma: Option<CmaTelemetry>,

    /// Ontology phase telemetry (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ontology: Option<OntologyTelemetry>,

    /// MEC (Molecular Emergent Computing) telemetry (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mec: Option<MecTelemetry>,

    /// Biomolecular adapter telemetry (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biomolecular: Option<BiomolecularTelemetry>,

    /// Materials adapter telemetry (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub materials: Option<MaterialsTelemetry>,

    /// GNN telemetry (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gnn: Option<GnnTelemetry>,

    /// Molecular dynamics telemetry (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub md: Option<MdTelemetry>,
}

impl TelemetryEvent {
    /// Creates a new telemetry event.
    pub fn new(phase: &str, metrics: HashMap<String, f64>, outcome: &PhaseOutcome) -> Self {
        Self {
            timestamp: Utc::now().to_rfc3339(),
            phase: phase.to_string(),
            metrics,
            outcome: format!("{:?}", outcome),
            rl_action: None,
            rl_reward: None,
            geometry: None,
            cma: None,
            ontology: None,
            mec: None,
            biomolecular: None,
            materials: None,
            gnn: None,
            md: None,
        }
    }

    /// Attaches geometry telemetry to this event.
    ///
    /// Should be called by orchestrator when geometry metrics are available.
    pub fn with_geometry(mut self, geom: &GeometryTelemetry) -> Self {
        self.geometry = Some(GeometryMetrics::from(geom));
        self
    }

    /// Attaches CMA-ES telemetry to this event.
    ///
    /// Should be called by orchestrator when CMA state is available.
    pub fn with_cma(mut self, cma_state: &CmaState) -> Self {
        self.cma = Some(CmaTelemetry::from(cma_state));
        self
    }

    /// Attaches ontology telemetry to this event.
    ///
    /// Should be called by Phase0 (Semantic Grounding) when ontology metrics are available.
    pub fn with_ontology(mut self, ontology: OntologyTelemetry) -> Self {
        self.ontology = Some(ontology);
        self
    }

    /// Attaches MEC telemetry to this event.
    ///
    /// Should be called by PhaseM (Molecular Emergent Computing) when MEC metrics are available.
    pub fn with_mec(mut self, mec: MecTelemetry) -> Self {
        self.mec = Some(mec);
        self
    }

    /// Attaches biomolecular adapter telemetry to this event.
    ///
    /// Should be called when biomolecular prediction metrics are available.
    pub fn with_biomolecular(mut self, biomolecular: BiomolecularTelemetry) -> Self {
        self.biomolecular = Some(biomolecular);
        self
    }

    /// Attaches materials adapter telemetry to this event.
    ///
    /// Should be called when materials prediction metrics are available.
    pub fn with_materials(mut self, materials: MaterialsTelemetry) -> Self {
        self.materials = Some(materials);
        self
    }

    /// Attaches GNN telemetry to this event.
    ///
    /// Should be called when graph neural network metrics are available.
    pub fn with_gnn(mut self, gnn: GnnTelemetry) -> Self {
        self.gnn = Some(gnn);
        self
    }

    /// Attaches molecular dynamics telemetry to this event.
    ///
    /// Should be called when MD simulation metrics are available.
    pub fn with_md(mut self, md: MdTelemetry) -> Self {
        self.md = Some(md);
        self
    }

    /// Serializes to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Writes to JSON file (appends).
    pub fn write_json(&self, path: &str) -> std::io::Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let json = self
            .to_json()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut file = OpenOptions::new().create(true).append(true).open(path)?;

        writeln!(file, "{}", json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_serialization() {
        let mut metrics = HashMap::new();
        metrics.insert("temperature".to_string(), 1.5);
        metrics.insert("entropy".to_string(), 0.75);

        let event = TelemetryEvent::new("Phase2", metrics, &PhaseOutcome::success());

        let json = event.to_json().unwrap();
        assert!(json.contains("Phase2"));
        assert!(json.contains("temperature"));
    }
}
