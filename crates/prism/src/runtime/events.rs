//! Event System - Event Sourcing for PRISM
//!
//! All state changes flow through events, enabling:
//! - Audit logging
//! - State replay
//! - Time-travel debugging
//! - Reactive UI updates

use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::broadcast;
use anyhow::Result;

/// Unique event identifier
pub type EventId = u64;

/// All events in the PRISM system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrismEvent {
    // ═══════════════════════════════════════════════════════════════════
    // Command Events (User/System initiated)
    // ═══════════════════════════════════════════════════════════════════

    /// Load a graph file
    LoadGraph { path: String },

    /// Load a protein structure
    LoadProtein { path: String },

    /// Start optimization
    StartOptimization { config: OptimizationConfig },

    /// Pause optimization
    PauseOptimization,

    /// Resume optimization
    ResumeOptimization,

    /// Stop optimization
    StopOptimization,

    /// Update parameter
    SetParameter { key: String, value: ParameterValue },

    // ═══════════════════════════════════════════════════════════════════
    // Pipeline Events (from PipelineActor)
    // ═══════════════════════════════════════════════════════════════════

    /// Graph loaded successfully
    GraphLoaded {
        vertices: usize,
        edges: usize,
        density: f64,
        estimated_chromatic: usize,
    },

    /// Phase started
    PhaseStarted {
        phase: PhaseId,
        name: String,
    },

    /// Phase progress update
    PhaseProgress {
        phase: PhaseId,
        iteration: usize,
        max_iterations: usize,
        colors: usize,
        conflicts: usize,
        temperature: f64,
    },

    /// Phase completed
    PhaseCompleted {
        phase: PhaseId,
        duration_ms: u64,
        final_colors: usize,
        final_conflicts: usize,
    },

    /// Phase failed
    PhaseFailed {
        phase: PhaseId,
        error: String,
    },

    /// New best solution found
    NewBestSolution {
        colors: usize,
        conflicts: usize,
        iteration: usize,
        phase: PhaseId,
    },

    /// Optimization completed
    OptimizationCompleted {
        total_duration_ms: u64,
        final_colors: usize,
        attempts: usize,
    },

    // ═══════════════════════════════════════════════════════════════════
    // GPU Events (from GpuActor)
    // ═══════════════════════════════════════════════════════════════════

    /// GPU status update
    GpuStatus {
        device_id: usize,
        name: String,
        utilization: f64,
        memory_used: u64,
        memory_total: u64,
        temperature: u32,
        power_watts: f32,
    },

    /// Kernel launched
    KernelLaunched {
        name: String,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    },

    /// Kernel completed
    KernelCompleted {
        name: String,
        duration_us: u64,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Thermodynamic Events (Phase 2)
    // ═══════════════════════════════════════════════════════════════════

    /// Replica state update
    ReplicaUpdate {
        replica_id: usize,
        temperature: f64,
        colors: usize,
        conflicts: usize,
        energy: f64,
    },

    /// Replica exchange occurred
    ReplicaExchange {
        replica_a: usize,
        replica_b: usize,
        accepted: bool,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Quantum Events (Phase 3)
    // ═══════════════════════════════════════════════════════════════════

    /// Quantum state update
    QuantumState {
        coherence: f64,
        top_amplitudes: Vec<(usize, f64)>, // (color, amplitude)
        tunneling_rate: f64,
    },

    /// Quantum measurement (collapse)
    QuantumMeasurement {
        measured_colors: usize,
        pre_collapse_entropy: f64,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Dendritic Events (Phase 0)
    // ═══════════════════════════════════════════════════════════════════

    /// Dendritic reservoir update
    DendriticUpdate {
        active_neurons: usize,
        total_neurons: usize,
        firing_rate: f64,
        pattern_detected: Option<String>,
    },

    // ═══════════════════════════════════════════════════════════════════
    // FluxNet RL Events
    // ═══════════════════════════════════════════════════════════════════

    /// RL action selected
    RlAction {
        state: String,
        action: String,
        q_value: f64,
        epsilon: f64,
    },

    /// RL reward received
    RlReward {
        reward: f64,
        cumulative_reward: f64,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Telemetry Events
    // ═══════════════════════════════════════════════════════════════════

    /// Metric recorded
    MetricRecorded {
        name: String,
        value: f64,
        labels: Vec<(String, String)>,
    },

    // ═══════════════════════════════════════════════════════════════════
    // LBS Events (Biomolecular Mode)
    // ═══════════════════════════════════════════════════════════════════

    /// Protein structure loaded
    ProteinLoaded {
        pdb_id: String,
        residues: usize,
        atoms: usize,
        chains: usize,
    },

    /// LBS phase started
    LbsPhaseStarted {
        phase: LbsPhaseId,
        name: String,
    },

    /// LBS phase progress
    LbsPhaseProgress {
        phase: LbsPhaseId,
        iteration: usize,
        max_iterations: usize,
        pockets_found: usize,
        best_druggability: f32,
    },

    /// LBS phase completed
    LbsPhaseCompleted {
        phase: LbsPhaseId,
        duration_ms: u64,
        pockets_found: usize,
    },

    /// Pocket detected
    PocketDetected {
        pocket_id: usize,
        volume: f32,
        druggability: f32,
        center: [f32; 3],
        residue_count: usize,
    },

    /// LBS prediction complete
    LbsPredictionComplete {
        total_pockets: usize,
        best_pocket_druggability: f32,
        total_duration_ms: u64,
        gpu_accelerated: bool,
    },

    /// GNN inference event
    GnnInference {
        num_nodes: usize,
        num_edges: usize,
        chromatic_prediction: usize,
        confidence: f32,
        gpu_used: bool,
        latency_ms: u64,
    },

    /// SASA computation event
    SasaComputed {
        num_atoms: usize,
        exposed_area: f32,
        buried_area: f32,
        latency_ms: u64,
    },

    // ═══════════════════════════════════════════════════════════════════
    // System Events
    // ═══════════════════════════════════════════════════════════════════

    /// Error occurred
    Error {
        source: String,
        message: String,
        recoverable: bool,
    },

    /// Shutdown requested
    Shutdown,
}

/// Phase identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhaseId {
    Phase0Dendritic,
    Phase1ActiveInference,
    Phase2Thermodynamic,
    Phase3Quantum,
    Phase4Geodesic,
    Phase5Memetic,
    Phase6Tda,
    Phase7Ensemble,
}

impl PhaseId {
    pub fn name(&self) -> &'static str {
        match self {
            PhaseId::Phase0Dendritic => "Dendritic Reservoir",
            PhaseId::Phase1ActiveInference => "Active Inference",
            PhaseId::Phase2Thermodynamic => "Thermodynamic",
            PhaseId::Phase3Quantum => "Quantum",
            PhaseId::Phase4Geodesic => "Geodesic",
            PhaseId::Phase5Memetic => "Memetic",
            PhaseId::Phase6Tda => "TDA",
            PhaseId::Phase7Ensemble => "Ensemble",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            PhaseId::Phase0Dendritic => 0,
            PhaseId::Phase1ActiveInference => 1,
            PhaseId::Phase2Thermodynamic => 2,
            PhaseId::Phase3Quantum => 3,
            PhaseId::Phase4Geodesic => 4,
            PhaseId::Phase5Memetic => 5,
            PhaseId::Phase6Tda => 6,
            PhaseId::Phase7Ensemble => 7,
        }
    }
}

/// LBS Phase identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LbsPhaseId {
    Phase0Surface,
    Phase1Beliefs,
    Phase2Sampling,
    Phase3Quantum,
    Phase4Cavity,
    Phase6Topology,
    WhcrRefinement,
}

impl LbsPhaseId {
    pub fn name(&self) -> &'static str {
        match self {
            LbsPhaseId::Phase0Surface => "Surface Reservoir",
            LbsPhaseId::Phase1Beliefs => "Pocket Beliefs",
            LbsPhaseId::Phase2Sampling => "MCMC Sampling",
            LbsPhaseId::Phase3Quantum => "Quantum Exploration",
            LbsPhaseId::Phase4Cavity => "Cavity Analysis",
            LbsPhaseId::Phase6Topology => "TDA Topology",
            LbsPhaseId::WhcrRefinement => "WHCR Refinement",
        }
    }
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub max_attempts: usize,
    pub target_colors: Option<usize>,
    pub enable_warmstart: bool,
    pub enable_fluxnet: bool,
    pub phases_enabled: Vec<PhaseId>,
    /// Enable AATGS async kernel scheduling
    pub enable_aatgs_async: bool,
    /// Enable multi-GPU parallel execution
    pub enable_multi_gpu: bool,
    /// Enable Ultra Kernel (fused 8-component GPU kernel)
    pub enable_ultra_kernel: bool,
    /// GPU device IDs for multi-GPU mode
    pub gpu_device_ids: Vec<usize>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            target_colors: None,
            enable_warmstart: true,
            enable_fluxnet: true,
            phases_enabled: vec![
                PhaseId::Phase0Dendritic,
                PhaseId::Phase1ActiveInference,
                PhaseId::Phase2Thermodynamic,
                PhaseId::Phase3Quantum,
                PhaseId::Phase4Geodesic,
                PhaseId::Phase6Tda,
                PhaseId::Phase7Ensemble,
            ],
            enable_aatgs_async: false,
            enable_multi_gpu: false,
            enable_ultra_kernel: false,
            gpu_device_ids: vec![0],
        }
    }
}

/// Parameter value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

/// Event bus for broadcasting events to all subscribers
pub struct EventBus {
    sender: broadcast::Sender<PrismEvent>,
    capacity: usize,
}

impl EventBus {
    /// Create a new event bus with given capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender, capacity }
    }

    /// Publish an event to all subscribers
    pub async fn publish(&self, event: PrismEvent) -> Result<()> {
        // Ignore send errors (no subscribers is OK)
        let _ = self.sender.send(event);
        Ok(())
    }

    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<PrismEvent> {
        self.sender.subscribe()
    }

    /// Get number of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_bus_pubsub() {
        let bus = EventBus::new(16);
        let mut rx = bus.subscribe();

        bus.publish(PrismEvent::GraphLoaded {
            vertices: 500,
            edges: 12500,
            density: 0.1,
            estimated_chromatic: 48,
        }).await.unwrap();

        let event = rx.recv().await.unwrap();
        match event {
            PrismEvent::GraphLoaded { vertices, .. } => assert_eq!(vertices, 500),
            _ => panic!("Wrong event type"),
        }
    }
}
