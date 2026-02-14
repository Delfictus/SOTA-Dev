//! State Store - Centralized Application State
//!
//! Thread-safe state management with:
//! - Lock-free ring buffers for time-series data
//! - Atomic updates for counters
//! - RwLock for complex state
//! - Efficient snapshots for UI rendering

use super::events::PhaseId;
use super::channels::RingBuffer;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Centralized state store
pub struct StateStore {
    /// Pipeline state
    pub pipeline: RwLock<PipelineState>,

    /// GPU state (per device)
    pub gpu: RwLock<Vec<GpuState>>,

    /// Phase states
    pub phases: RwLock<Vec<PhaseState>>,

    /// Graph state
    pub graph: RwLock<GraphState>,

    /// Optimization state
    pub optimization: RwLock<OptimizationState>,

    /// Time-series data (lock-free ring buffers)
    pub convergence_history: RingBuffer<ConvergencePoint>,
    pub gpu_utilization_history: RingBuffer<GpuUtilPoint>,
    pub temperature_history: RingBuffer<TemperaturePoint>,
    pub conflict_history: RingBuffer<ConflictPoint>,

    /// Atomic counters (lock-free)
    pub total_iterations: AtomicU64,
    pub total_kernel_launches: AtomicU64,
    pub events_processed: AtomicU64,

    /// Start time for uptime tracking
    start_time: Instant,
}

impl StateStore {
    /// Create a new state store
    pub fn new(ring_buffer_size: usize) -> Self {
        Self {
            pipeline: RwLock::new(PipelineState::default()),
            gpu: RwLock::new(vec![GpuState::default()]),
            phases: RwLock::new(Self::init_phases()),
            graph: RwLock::new(GraphState::default()),
            optimization: RwLock::new(OptimizationState::default()),
            convergence_history: RingBuffer::new(ring_buffer_size),
            gpu_utilization_history: RingBuffer::new(ring_buffer_size),
            temperature_history: RingBuffer::new(ring_buffer_size),
            conflict_history: RingBuffer::new(ring_buffer_size),
            total_iterations: AtomicU64::new(0),
            total_kernel_launches: AtomicU64::new(0),
            events_processed: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    fn init_phases() -> Vec<PhaseState> {
        vec![
            PhaseState::new(PhaseId::Phase0Dendritic),
            PhaseState::new(PhaseId::Phase1ActiveInference),
            PhaseState::new(PhaseId::Phase2Thermodynamic),
            PhaseState::new(PhaseId::Phase3Quantum),
            PhaseState::new(PhaseId::Phase4Geodesic),
            PhaseState::new(PhaseId::Phase6Tda),
            PhaseState::new(PhaseId::Phase7Ensemble),
        ]
    }

    /// Take a snapshot of all state for UI rendering
    pub async fn snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            pipeline: self.pipeline.read().clone(),
            gpu: self.gpu.read().clone(),
            phases: self.phases.read().clone(),
            graph: self.graph.read().clone(),
            optimization: self.optimization.read().clone(),
            convergence_history: self.convergence_history.to_vec(),
            gpu_utilization_history: self.gpu_utilization_history.to_vec(),
            temperature_history: self.temperature_history.to_vec(),
            conflict_history: self.conflict_history.to_vec(),
            total_iterations: self.total_iterations.load(Ordering::Relaxed),
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }

    /// Record a convergence data point
    pub fn record_convergence(&self, colors: usize, conflicts: usize, iteration: u64) {
        self.convergence_history.push(ConvergencePoint {
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
            iteration,
            colors,
            conflicts,
        });
    }

    /// Record GPU utilization
    pub fn record_gpu_util(&self, device_id: usize, utilization: f64, memory_pct: f64) {
        self.gpu_utilization_history.push(GpuUtilPoint {
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
            device_id,
            utilization,
            memory_pct,
        });
    }

    /// Record temperature
    pub fn record_temperature(&self, replica_id: usize, temperature: f64) {
        self.temperature_history.push(TemperaturePoint {
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
            replica_id,
            temperature,
        });
    }

    /// Increment iteration counter
    pub fn inc_iterations(&self) {
        self.total_iterations.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment kernel launch counter
    pub fn inc_kernel_launches(&self) {
        self.total_kernel_launches.fetch_add(1, Ordering::Relaxed);
    }
}

/// Complete state snapshot for UI rendering
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub pipeline: PipelineState,
    pub gpu: Vec<GpuState>,
    pub phases: Vec<PhaseState>,
    pub graph: GraphState,
    pub optimization: OptimizationState,
    pub convergence_history: Vec<ConvergencePoint>,
    pub gpu_utilization_history: Vec<GpuUtilPoint>,
    pub temperature_history: Vec<TemperaturePoint>,
    pub conflict_history: Vec<ConflictPoint>,
    pub total_iterations: u64,
    pub uptime_secs: u64,
}

/// Pipeline execution state
#[derive(Debug, Clone, Default)]
pub struct PipelineState {
    pub status: PipelineStatus,
    pub current_phase: Option<PhaseId>,
    pub current_attempt: usize,
    pub max_attempts: usize,
    pub start_time: Option<u64>, // Unix timestamp ms
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PipelineStatus {
    #[default]
    Idle,
    Loading,
    Running,
    Paused,
    Completed,
    Failed,
}

/// GPU device state
#[derive(Debug, Clone)]
pub struct GpuState {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub utilization: f64,
    pub memory_used: u64,
    pub memory_total: u64,
    pub temperature: u32,
    pub power_watts: f32,
    pub active_kernels: Vec<ActiveKernel>,
}

impl Default for GpuState {
    fn default() -> Self {
        Self {
            device_id: 0,
            name: "Unknown GPU".into(),
            compute_capability: (8, 6),
            utilization: 0.0,
            memory_used: 0,
            memory_total: 12 * 1024 * 1024 * 1024,
            temperature: 45,
            power_watts: 0.0,
            active_kernels: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActiveKernel {
    pub name: String,
    pub started_ms: u64,
    pub progress: f64,
}

/// Phase execution state
#[derive(Debug, Clone)]
pub struct PhaseState {
    pub id: PhaseId,
    pub name: String,
    pub status: PhaseStatus,
    pub progress: f64,
    pub iteration: usize,
    pub max_iterations: usize,
    pub start_time: Option<u64>,
    pub duration_ms: u64,
    pub metrics: PhaseMetrics,
}

impl PhaseState {
    pub fn new(id: PhaseId) -> Self {
        Self {
            id,
            name: id.name().into(),
            status: PhaseStatus::Pending,
            progress: 0.0,
            iteration: 0,
            max_iterations: 0,
            start_time: None,
            duration_ms: 0,
            metrics: PhaseMetrics::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PhaseStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// Phase-specific metrics
#[derive(Debug, Clone, Default)]
pub struct PhaseMetrics {
    pub colors: usize,
    pub conflicts: usize,
    pub temperature: f64,
    pub energy: f64,
    pub acceptance_rate: f64,
    pub improvement_rate: f64,
}

/// Graph state
#[derive(Debug, Clone, Default)]
pub struct GraphState {
    pub loaded: bool,
    pub path: Option<String>,
    pub vertices: usize,
    pub edges: usize,
    pub density: f64,
    pub max_degree: usize,
    pub avg_degree: f64,
    pub estimated_chromatic: usize,
    pub clique_lower_bound: usize,
}

/// Optimization state
#[derive(Debug, Clone, Default)]
pub struct OptimizationState {
    pub current_colors: usize,
    pub current_conflicts: usize,
    pub best_colors: usize,
    pub best_conflicts: usize,
    pub best_iteration: u64,
    pub best_phase: Option<PhaseId>,
    pub target_colors: Option<usize>,

    // Replica states (for Phase 2)
    pub replicas: Vec<ReplicaState>,

    // Quantum state (for Phase 3)
    pub quantum: QuantumState,

    // Dendritic state (for Phase 0)
    pub dendritic: DendriticState,

    // FluxNet RL state
    pub fluxnet: FluxNetState,
}

#[derive(Debug, Clone)]
pub struct ReplicaState {
    pub id: usize,
    pub temperature: f64,
    pub colors: usize,
    pub conflicts: usize,
    pub energy: f64,
    pub is_best: bool,
}

#[derive(Debug, Clone, Default)]
pub struct QuantumState {
    pub coherence: f64,
    pub tunneling_rate: f64,
    pub top_amplitudes: Vec<(usize, f64)>, // (color_count, amplitude)
    pub entropy: f64,
}

#[derive(Debug, Clone, Default)]
pub struct DendriticState {
    pub active_neurons: usize,
    pub total_neurons: usize,
    pub firing_rate: f64,
    pub compartment_activity: [f64; 4],
    pub detected_patterns: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct FluxNetState {
    pub epsilon: f64,
    pub cumulative_reward: f64,
    pub last_action: Option<String>,
    pub last_q_value: f64,
    pub exploration_count: usize,
    pub exploitation_count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Time-Series Data Points (stored in ring buffers)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub struct ConvergencePoint {
    pub timestamp_ms: u64,
    pub iteration: u64,
    pub colors: usize,
    pub conflicts: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuUtilPoint {
    pub timestamp_ms: u64,
    pub device_id: usize,
    pub utilization: f64,
    pub memory_pct: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TemperaturePoint {
    pub timestamp_ms: u64,
    pub replica_id: usize,
    pub temperature: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ConflictPoint {
    pub timestamp_ms: u64,
    pub conflicts: usize,
    pub delta: i32,
}
