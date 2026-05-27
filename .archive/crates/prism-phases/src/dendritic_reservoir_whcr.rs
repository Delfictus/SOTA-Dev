//! PRISM Dendritic Reservoir for WHCR (DR-WHCR)
//!
//! Live neuromorphic co-processor that evolves WITH the repair process,
//! providing real-time adaptive guidance for conflict resolution.
//!
//! # Architecture
//!
//! ```text
//! WHCR Iteration t
//!        ↓
//! [Conflict State] → [DR Input Layer] → [Dendritic Compartments] → [Output Layer]
//!        ↑                                      ↓
//!        └──────────── [Priority Modulation] ←──┘
//!                             ↓
//!                     WHCR Iteration t+1
//! ```
//!
//! # Key Innovations
//!
//! 1. **Dynamic reservoir topology** mapping to current conflict structure
//! 2. **Multi-compartment dendritic processing** with varied time constants
//! 3. **Conflict momentum tracking** to identify stubborn vs transient conflicts
//! 4. **Cascade prediction** via reservoir echo state dynamics
//! 5. **Temporal memory** preventing oscillation and cycling
//!
//! # Patent Claims
//!
//! - "Neuromorphic co-processor for combinatorial optimization"
//! - "Dendritic reservoir computing for dynamic conflict resolution"  
//! - "Real-time neuromorphic guidance in hierarchical graph repair"

use parking_lot::RwLock;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Number of dendritic compartments per vertex
pub const NUM_COMPARTMENTS: usize = 4;

/// Compartment indices
pub const COMPARTMENT_PROXIMAL: usize = 0; // Fast dynamics (tau ~ 1 iteration)
pub const COMPARTMENT_DISTAL_1: usize = 1; // Medium dynamics (tau ~ 5 iterations)
pub const COMPARTMENT_DISTAL_2: usize = 2; // Slow dynamics (tau ~ 20 iterations)
pub const COMPARTMENT_SPINE: usize = 3; // Very slow (tau ~ 50 iterations)

/// Output signal indices
pub const OUTPUT_PRIORITY_MOD: usize = 0; // Priority modulation for WHCR
pub const OUTPUT_CASCADE_PRED: usize = 1; // Cascade potential prediction
pub const OUTPUT_RECEPTIVITY: usize = 2; // Repair receptivity score
pub const OUTPUT_MOMENTUM: usize = 3; // Conflict momentum indicator
pub const NUM_OUTPUTS: usize = 4;

/// Time constants (decay factors) for each compartment
pub const TAU_DECAY: [f32; NUM_COMPARTMENTS] = [
    0.1,  // Proximal: rapid response, 90% decay per step
    0.5,  // Distal 1: medium memory
    0.85, // Distal 2: slow integration
    0.95, // Spine: long-term conflict memory
];

// ═══════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

/// Dendritic compartment state
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DendriticCompartment {
    pub activation: f32,
    pub calcium: f32,
    pub threshold: f32,
    pub refractory: f32,
}

/// Per-vertex dendritic state
#[derive(Debug, Clone)]
#[repr(C)]
pub struct VertexDendriticState {
    pub compartments: [DendriticCompartment; NUM_COMPARTMENTS],
    pub soma_potential: f32,
    pub spike_history: f32,
    pub conflict_memory: f32,
    pub last_repair_iteration: i32,
}

impl Default for VertexDendriticState {
    fn default() -> Self {
        Self {
            compartments: [DendriticCompartment::default(); NUM_COMPARTMENTS],
            soma_potential: 0.0,
            spike_history: 0.0,
            conflict_memory: 0.0,
            last_repair_iteration: -100,
        }
    }
}

/// Reservoir connection (sparse)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ReservoirConnection {
    pub source_vertex: i32,
    pub target_vertex: i32,
    pub source_compartment: i32,
    pub target_compartment: i32,
    pub weight: f32,
    pub delay: f32,
}

/// Reservoir output signals for a single vertex
#[derive(Debug, Clone, Copy, Default)]
pub struct ReservoirOutputs {
    pub priority_modulation: f32,
    pub cascade_prediction: f32,
    pub repair_receptivity: f32,
    pub conflict_momentum: f32,
}

/// Pattern detection results
#[derive(Debug, Clone, Default)]
pub struct PatternAnalysis {
    pub oscillating_vertices: Vec<u32>,
    pub stubborn_vertices: Vec<u32>,
    pub improving_vertices: Vec<u32>,
    pub cascade_candidates: Vec<u32>,
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Dendritic reservoir configuration
#[derive(Debug, Clone)]
pub struct DendriticReservoirConfig {
    /// Spectral radius for echo state property
    pub spectral_radius: f32,
    /// Input scaling factor
    pub input_scaling: f32,
    /// Reservoir state leak rate
    pub leak_rate: f32,
    /// Connection sparsity (0.0 to 1.0)
    pub sparsity: f32,
    /// Conflict history length (iterations)
    pub history_length: usize,
    /// How much reservoir affects WHCR priorities (0.0 to 1.0)
    pub reservoir_influence: f32,
    /// Online learning rate
    pub learning_rate: f32,
    /// Oscillation detection threshold
    pub oscillation_threshold: f32,
    /// Stubborn conflict threshold
    pub stubborn_threshold: f32,
    /// Enable online learning
    pub enable_learning: bool,
    /// Enable pattern detection
    pub enable_pattern_detection: bool,
}

impl Default for DendriticReservoirConfig {
    fn default() -> Self {
        Self {
            spectral_radius: 0.9,
            input_scaling: 0.3,
            leak_rate: 0.3,
            sparsity: 0.1,
            history_length: 50,
            reservoir_influence: 0.3,
            learning_rate: 0.001,
            oscillation_threshold: 0.5,
            stubborn_threshold: 0.1,
            enable_learning: true,
            enable_pattern_detection: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WHCR ITERATION STATE (input to reservoir)
// ═══════════════════════════════════════════════════════════════════════════

/// State from a WHCR iteration, used as reservoir input
#[derive(Debug, Clone)]
pub struct WHCRIterationState {
    pub conflict_counts: Vec<f32>,
    pub conflict_deltas: Vec<f32>,
    pub colors: Vec<i32>,
    pub moves_applied: Vec<i32>,
    pub wavelet_details: Option<Vec<f32>>,
    pub num_colors: usize,
    pub iteration: i32,
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU IMPLEMENTATION (for testing and fallback)
// ═══════════════════════════════════════════════════════════════════════════

/// CPU-based dendritic reservoir implementation
pub struct DendriticReservoirCPU {
    config: DendriticReservoirConfig,
    num_vertices: usize,

    // Vertex states
    vertex_states: Vec<VertexDendriticState>,

    // Sparse connections (CSR format)
    connection_row_ptr: Vec<usize>,
    connections: Vec<ReservoirConnection>,

    // Weights
    input_weights: Vec<f32>,  // [num_vertices * NUM_COMPARTMENTS]
    output_weights: Vec<f32>, // [num_vertices * NUM_OUTPUTS]

    // Outputs
    outputs: Vec<f32>, // [num_vertices * NUM_OUTPUTS]

    // Conflict history (ring buffer)
    conflict_history: Vec<f32>, // [history_length * num_vertices]
    history_index: usize,

    // Global state
    iteration_count: usize,
    global_activity: f32,
    conflict_trend: f32,
}

impl DendriticReservoirCPU {
    /// Create new reservoir from graph structure
    pub fn new(
        num_vertices: usize,
        graph_row_ptr: &[usize],
        graph_col_idx: &[u32],
        initial_conflicts: &[f32],
        config: DendriticReservoirConfig,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize vertex states
        let mut vertex_states = Vec::with_capacity(num_vertices);
        for v in 0..num_vertices {
            let mut state = VertexDendriticState::default();
            state.conflict_memory = initial_conflicts[v];
            for c in 0..NUM_COMPARTMENTS {
                state.compartments[c].activation = rng.gen::<f32>() * 0.1;
                state.compartments[c].threshold = 0.5 + rng.gen::<f32>() * 0.2;
            }
            vertex_states.push(state);
        }

        // Initialize connections based on graph structure
        let mut connection_row_ptr = vec![0usize; num_vertices + 1];
        let mut connections = Vec::new();

        for v in 0..num_vertices {
            let start = graph_row_ptr[v];
            let end = graph_row_ptr[v + 1];
            let degree = end - start;

            connection_row_ptr[v] = connections.len();

            for i in start..end {
                let neighbor = graph_col_idx[i] as usize;

                // Probabilistic connection based on conflict weight
                let conflict_weight = initial_conflicts[neighbor] + 0.1;
                let connection_prob = config.sparsity * conflict_weight;

                if rng.gen::<f32>() < connection_prob {
                    let src_comp = rng.gen_range(0..NUM_COMPARTMENTS as i32);
                    let tgt_comp = rng.gen_range(0..NUM_COMPARTMENTS as i32);
                    let weight = (rng.gen::<f32>() * 2.0 - 1.0) * config.spectral_radius
                        / (degree as f32).sqrt();

                    connections.push(ReservoirConnection {
                        source_vertex: v as i32,
                        target_vertex: neighbor as i32,
                        source_compartment: src_comp,
                        target_compartment: tgt_comp,
                        weight,
                        delay: rng.gen::<f32>() * 3.0,
                    });
                }
            }
        }
        connection_row_ptr[num_vertices] = connections.len();

        // Initialize input weights
        let mut input_weights = Vec::with_capacity(num_vertices * NUM_COMPARTMENTS);
        for _ in 0..num_vertices {
            for c in 0..NUM_COMPARTMENTS {
                let base = match c {
                    COMPARTMENT_PROXIMAL => 1.5,
                    COMPARTMENT_DISTAL_1 => 1.0,
                    COMPARTMENT_DISTAL_2 => 0.5,
                    COMPARTMENT_SPINE => 0.2,
                    _ => 1.0,
                };
                input_weights.push(config.input_scaling * base * (0.8 + rng.gen::<f32>() * 0.4));
            }
        }

        // Initialize output weights (identity for now)
        let output_weights = vec![1.0f32; num_vertices * NUM_OUTPUTS];

        // Initialize outputs
        let outputs = vec![0.0f32; num_vertices * NUM_OUTPUTS];

        // Initialize history buffer
        let conflict_history = vec![0.0f32; config.history_length * num_vertices];

        Self {
            config,
            num_vertices,
            vertex_states,
            connection_row_ptr,
            connections,
            input_weights,
            output_weights,
            outputs,
            conflict_history,
            history_index: 0,
            iteration_count: 0,
            global_activity: 0.0,
            conflict_trend: 0.0,
        }
    }

    /// Process one WHCR iteration through the reservoir
    pub fn step(&mut self, whcr_state: &WHCRIterationState) {
        // 1. Process input through dendritic compartments
        self.process_dendritic_input(whcr_state);

        // 2. Process recurrent connections
        self.process_recurrent_connections();

        // 3. Compute soma integration
        self.compute_soma_integration();

        // 4. Compute outputs
        self.compute_outputs(whcr_state.iteration);

        // 5. Update conflict history
        self.update_history(&whcr_state.conflict_counts);

        self.iteration_count += 1;
    }

    fn process_dendritic_input(&mut self, whcr_state: &WHCRIterationState) {
        for v in 0..self.num_vertices {
            let state = &mut self.vertex_states[v];

            let conflict_signal = whcr_state.conflict_counts[v];
            let delta_signal = whcr_state.conflict_deltas[v];
            let wavelet_signal = whcr_state
                .wavelet_details
                .as_ref()
                .map(|w| w[v])
                .unwrap_or(0.0);

            let just_repaired = whcr_state.moves_applied[v] != 0;
            if just_repaired {
                state.last_repair_iteration = whcr_state.iteration;
            }

            for c in 0..NUM_COMPARTMENTS {
                let comp = &mut state.compartments[c];
                let tau = TAU_DECAY[c];
                let w = self.input_weights[v * NUM_COMPARTMENTS + c];

                // Decay existing activation
                comp.activation *= tau;

                // Compute input contribution based on compartment type
                let input_contribution = match c {
                    COMPARTMENT_PROXIMAL => w * conflict_signal,
                    COMPARTMENT_DISTAL_1 => w * (conflict_signal + delta_signal * 2.0),
                    COMPARTMENT_DISTAL_2 => w * (conflict_signal + wavelet_signal.abs()),
                    COMPARTMENT_SPINE => {
                        comp.calcium += conflict_signal * 0.01;
                        comp.calcium = comp.calcium.min(1.0);
                        w * conflict_signal * 0.3
                    }
                    _ => 0.0,
                };

                if comp.refractory <= 0.0 {
                    comp.activation += input_contribution;
                } else {
                    comp.refractory -= 1.0;
                }

                // Homeostatic threshold adaptation
                comp.threshold += 0.001 * (comp.activation - comp.threshold);
                comp.threshold = comp.threshold.clamp(0.3, 0.9);
            }

            // Update conflict memory
            state.conflict_memory = 0.9 * state.conflict_memory + 0.1 * conflict_signal;
        }
    }

    fn process_recurrent_connections(&mut self) {
        // Accumulate recurrent input
        let mut recurrent_input = vec![[0.0f32; NUM_COMPARTMENTS]; self.num_vertices];

        for conn in &self.connections {
            let source_act = self.vertex_states[conn.source_vertex as usize].compartments
                [conn.source_compartment as usize]
                .activation;

            let transmitted = source_act.tanh() * conn.weight;
            recurrent_input[conn.target_vertex as usize][conn.target_compartment as usize] +=
                transmitted;
        }

        // Apply recurrent input with leak rate
        for v in 0..self.num_vertices {
            for c in 0..NUM_COMPARTMENTS {
                let state = &mut self.vertex_states[v];
                state.compartments[c].activation = (1.0 - self.config.leak_rate)
                    * state.compartments[c].activation
                    + self.config.leak_rate * recurrent_input[v][c].tanh();
            }
        }
    }

    fn compute_soma_integration(&mut self) {
        let weights = [1.0f32, 0.6, 0.3, 0.1];

        for v in 0..self.num_vertices {
            let state = &mut self.vertex_states[v];

            let mut soma_input = 0.0;
            for c in 0..NUM_COMPARTMENTS {
                soma_input += weights[c] * state.compartments[c].activation;
            }

            // Leaky integrate
            state.soma_potential = 0.8 * state.soma_potential + 0.2 * soma_input;

            // Check for spike
            let threshold = 0.5;
            if state.soma_potential > threshold {
                state.spike_history = 0.9 * state.spike_history + 0.1;
                state.soma_potential = 0.0;
                state.compartments[COMPARTMENT_PROXIMAL].refractory = 2.0;
            } else {
                state.spike_history *= 0.95;
            }
        }
    }

    fn compute_outputs(&mut self, iteration: i32) {
        for v in 0..self.num_vertices {
            let state = &self.vertex_states[v];
            let repair_age = iteration - state.last_repair_iteration;

            // OUTPUT 0: Priority Modulation
            let mut priority_mod = state.compartments[COMPARTMENT_PROXIMAL].activation * 2.0
                + state.compartments[COMPARTMENT_DISTAL_2].activation * 1.5
                + state.compartments[COMPARTMENT_SPINE].calcium * 3.0;

            if repair_age > 10 {
                priority_mod *= 1.2;
            }
            self.outputs[v * NUM_OUTPUTS + OUTPUT_PRIORITY_MOD] = priority_mod;

            // OUTPUT 1: Cascade Prediction
            let cascade_potential = state.soma_potential * (1.0 - state.spike_history)
                + state.compartments[COMPARTMENT_DISTAL_1].activation * 0.5;
            self.outputs[v * NUM_OUTPUTS + OUTPUT_CASCADE_PRED] = cascade_potential;

            // OUTPUT 2: Repair Receptivity
            let mut receptivity = 1.0 - state.spike_history;
            if repair_age < 5 {
                receptivity *= 0.3;
            }
            if state.conflict_memory < state.compartments[COMPARTMENT_SPINE].calcium {
                receptivity *= 1.2;
            }
            self.outputs[v * NUM_OUTPUTS + OUTPUT_RECEPTIVITY] = receptivity.clamp(0.0, 1.0);

            // OUTPUT 3: Conflict Momentum
            let momentum = state.compartments[COMPARTMENT_PROXIMAL].activation
                - state.compartments[COMPARTMENT_DISTAL_2].activation;
            self.outputs[v * NUM_OUTPUTS + OUTPUT_MOMENTUM] = momentum;
        }
    }

    fn update_history(&mut self, conflict_counts: &[f32]) {
        let offset = self.history_index * self.num_vertices;
        for v in 0..self.num_vertices {
            self.conflict_history[offset + v] = conflict_counts[v];
        }
        self.history_index = (self.history_index + 1) % self.config.history_length;
    }

    /// Get reservoir outputs for a vertex
    pub fn get_outputs(&self, vertex: usize) -> ReservoirOutputs {
        ReservoirOutputs {
            priority_modulation: self.outputs[vertex * NUM_OUTPUTS + OUTPUT_PRIORITY_MOD],
            cascade_prediction: self.outputs[vertex * NUM_OUTPUTS + OUTPUT_CASCADE_PRED],
            repair_receptivity: self.outputs[vertex * NUM_OUTPUTS + OUTPUT_RECEPTIVITY],
            conflict_momentum: self.outputs[vertex * NUM_OUTPUTS + OUTPUT_MOMENTUM],
        }
    }

    /// Modulate WHCR wavelet priorities with reservoir outputs
    pub fn modulate_priorities(&self, wavelet_priorities: &[f32]) -> Vec<f32> {
        let mut final_priorities = Vec::with_capacity(self.num_vertices);

        for v in 0..self.num_vertices {
            let wavelet_p = wavelet_priorities[v];

            // Skip non-conflicting vertices
            if wavelet_p < 0.0 {
                final_priorities.push(wavelet_p);
                continue;
            }

            let outputs = self.get_outputs(v);

            // Compute reservoir-based adjustment
            let reservoir_score = outputs.priority_modulation * 1.0
                + outputs.cascade_prediction * 1.5
                + outputs.repair_receptivity * 0.5
                + outputs.conflict_momentum * 0.3;

            // Blend wavelet and reservoir priorities
            let blended = (1.0 - self.config.reservoir_influence) * wavelet_p
                + self.config.reservoir_influence * (wavelet_p + reservoir_score);

            final_priorities.push(blended);
        }

        final_priorities
    }

    /// Detect patterns in conflict dynamics
    pub fn detect_patterns(&self) -> PatternAnalysis {
        let mut analysis = PatternAnalysis::default();

        if !self.config.enable_pattern_detection {
            return analysis;
        }

        for v in 0..self.num_vertices {
            // Detect oscillations
            let oscillation_score = self.compute_oscillation_score(v);
            if oscillation_score > self.config.oscillation_threshold {
                analysis.oscillating_vertices.push(v as u32);
            }

            // Detect stubborn conflicts
            let stubborn_score = self.compute_stubborn_score(v);
            if stubborn_score > 0.7 {
                analysis.stubborn_vertices.push(v as u32);
            }

            // Detect improving vertices
            let outputs = self.get_outputs(v);
            if outputs.conflict_momentum < -0.3 {
                analysis.improving_vertices.push(v as u32);
            }

            // Detect cascade candidates
            if outputs.cascade_prediction > 0.5 && outputs.repair_receptivity > 0.6 {
                analysis.cascade_candidates.push(v as u32);
            }
        }

        analysis
    }

    fn compute_oscillation_score(&self, vertex: usize) -> f32 {
        let mut sign_changes = 0;
        let mut prev_delta = 0.0f32;

        for i in 1..self.config.history_length {
            let idx_curr = ((self.history_index + self.config.history_length - i)
                % self.config.history_length)
                * self.num_vertices
                + vertex;
            let idx_prev = ((self.history_index + self.config.history_length - i - 1)
                % self.config.history_length)
                * self.num_vertices
                + vertex;

            let curr_delta = self.conflict_history[idx_curr] - self.conflict_history[idx_prev];

            if i > 1 && prev_delta * curr_delta < 0.0 {
                sign_changes += 1;
            }
            prev_delta = curr_delta;
        }

        sign_changes as f32 / (self.config.history_length - 2) as f32
    }

    fn compute_stubborn_score(&self, vertex: usize) -> f32 {
        let mut conflict_count = 0;

        for i in 0..self.config.history_length {
            let idx = i * self.num_vertices + vertex;
            if self.conflict_history[idx] > self.config.stubborn_threshold {
                conflict_count += 1;
            }
        }

        let persistence = conflict_count as f32 / self.config.history_length as f32;
        let calcium = self.vertex_states[vertex].compartments[COMPARTMENT_SPINE].calcium;

        0.6 * persistence + 0.4 * calcium
    }

    /// Update weights based on successful repairs (online learning)
    pub fn update_weights(&mut self, successful_repairs: &[bool]) {
        if !self.config.enable_learning {
            return;
        }

        for conn in &mut self.connections {
            let source_act = self.vertex_states[conn.source_vertex as usize].compartments
                [conn.source_compartment as usize]
                .activation;
            let target_act = self.vertex_states[conn.target_vertex as usize].compartments
                [conn.target_compartment as usize]
                .activation;
            let target_success = successful_repairs[conn.target_vertex as usize];

            if target_success {
                // LTP: strengthen connection
                let delta = self.config.learning_rate * source_act * target_act;
                conn.weight += delta;
            } else if source_act > 0.5 && target_act > 0.5 {
                // LTD: weaken if both active but no success
                conn.weight -= self.config.learning_rate * 0.1;
            }

            conn.weight = conn.weight.clamp(-1.0, 1.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU IMPLEMENTATION WRAPPER
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
pub mod gpu {
    use super::*;

    // FFI declarations for CUDA kernels
    extern "C" {
        fn launch_init_reservoir(
            reservoir: *mut std::ffi::c_void,
            graph_row_ptr: *const i32,
            graph_col_idx: *const i32,
            initial_conflicts: *const f32,
            num_vertices: i32,
            seed: u64,
            stream: *mut std::ffi::c_void,
        );

        fn launch_reservoir_step(
            reservoir: *mut std::ffi::c_void,
            whcr_state: *const std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        );

        fn launch_get_modulated_priorities(
            reservoir: *const std::ffi::c_void,
            wavelet_priorities: *const f32,
            final_priorities: *mut f32,
            reservoir_influence: f32,
            stream: *mut std::ffi::c_void,
        );

        fn launch_pattern_detection(
            reservoir: *const std::ffi::c_void,
            oscillation_scores: *mut f32,
            stubborn_scores: *mut f32,
            stubborn_threshold: f32,
            stream: *mut std::ffi::c_void,
        );

        fn launch_weight_update(
            reservoir: *mut std::ffi::c_void,
            successful_repairs: *const i32,
            learning_rate: f32,
            stream: *mut std::ffi::c_void,
        );
    }

    /// GPU-accelerated dendritic reservoir
    pub struct DendriticReservoirGPU {
        // GPU memory handles would go here
        config: DendriticReservoirConfig,
        num_vertices: usize,
        // ... device pointers
    }

    impl DendriticReservoirGPU {
        pub fn new(
            _num_vertices: usize,
            _graph_row_ptr: &[usize],
            _graph_col_idx: &[u32],
            _initial_conflicts: &[f32],
            config: DendriticReservoirConfig,
        ) -> Self {
            // Would allocate GPU memory and call launch_init_reservoir
            todo!("GPU implementation requires CUDA runtime")
        }

        pub fn step(&mut self, _whcr_state: &WHCRIterationState) {
            // Would call launch_reservoir_step
            todo!("GPU implementation requires CUDA runtime")
        }

        pub fn modulate_priorities(&self, _wavelet_priorities: &[f32]) -> Vec<f32> {
            // Would call launch_get_modulated_priorities
            todo!("GPU implementation requires CUDA runtime")
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED INTERFACE
// ═══════════════════════════════════════════════════════════════════════════

/// Backend selection for dendritic reservoir
pub enum ReservoirBackend {
    CPU,
    #[cfg(feature = "cuda")]
    GPU,
}

/// Unified dendritic reservoir interface
pub enum DendriticReservoir {
    CPU(DendriticReservoirCPU),
    #[cfg(feature = "cuda")]
    GPU(gpu::DendriticReservoirGPU),
}

impl DendriticReservoir {
    pub fn new(
        num_vertices: usize,
        graph_row_ptr: &[usize],
        graph_col_idx: &[u32],
        initial_conflicts: &[f32],
        config: DendriticReservoirConfig,
        backend: ReservoirBackend,
    ) -> Self {
        match backend {
            ReservoirBackend::CPU => DendriticReservoir::CPU(DendriticReservoirCPU::new(
                num_vertices,
                graph_row_ptr,
                graph_col_idx,
                initial_conflicts,
                config,
            )),
            #[cfg(feature = "cuda")]
            ReservoirBackend::GPU => DendriticReservoir::GPU(gpu::DendriticReservoirGPU::new(
                num_vertices,
                graph_row_ptr,
                graph_col_idx,
                initial_conflicts,
                config,
            )),
        }
    }

    pub fn step(&mut self, whcr_state: &WHCRIterationState) {
        match self {
            DendriticReservoir::CPU(r) => r.step(whcr_state),
            #[cfg(feature = "cuda")]
            DendriticReservoir::GPU(r) => r.step(whcr_state),
        }
    }

    pub fn modulate_priorities(&self, wavelet_priorities: &[f32]) -> Vec<f32> {
        match self {
            DendriticReservoir::CPU(r) => r.modulate_priorities(wavelet_priorities),
            #[cfg(feature = "cuda")]
            DendriticReservoir::GPU(r) => r.modulate_priorities(wavelet_priorities),
        }
    }

    pub fn detect_patterns(&self) -> PatternAnalysis {
        match self {
            DendriticReservoir::CPU(r) => r.detect_patterns(),
            #[cfg(feature = "cuda")]
            DendriticReservoir::GPU(_) => {
                // GPU pattern detection would need separate implementation
                PatternAnalysis::default()
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> (Vec<usize>, Vec<u32>) {
        // Simple triangle: 0-1, 1-2, 2-0
        let row_ptr = vec![0, 2, 4, 6];
        let col_idx = vec![1, 2, 0, 2, 0, 1];
        (row_ptr, col_idx)
    }

    #[test]
    fn test_reservoir_creation() {
        let (row_ptr, col_idx) = create_test_graph();
        let initial_conflicts = vec![1.0, 1.0, 1.0];
        let config = DendriticReservoirConfig::default();

        let reservoir =
            DendriticReservoirCPU::new(3, &row_ptr, &col_idx, &initial_conflicts, config);

        assert_eq!(reservoir.num_vertices, 3);
        assert_eq!(reservoir.vertex_states.len(), 3);
    }

    #[test]
    fn test_reservoir_step() {
        let (row_ptr, col_idx) = create_test_graph();
        let initial_conflicts = vec![1.0, 1.0, 1.0];
        let config = DendriticReservoirConfig::default();

        let mut reservoir =
            DendriticReservoirCPU::new(3, &row_ptr, &col_idx, &initial_conflicts, config);

        let whcr_state = WHCRIterationState {
            conflict_counts: vec![1.0, 0.5, 0.0],
            conflict_deltas: vec![-0.5, -0.5, -1.0],
            colors: vec![0, 1, 2],
            moves_applied: vec![0, 1, 1],
            wavelet_details: Some(vec![0.5, 0.3, 0.1]),
            num_colors: 3,
            iteration: 1,
        };

        reservoir.step(&whcr_state);

        // Verify outputs were computed
        let outputs = reservoir.get_outputs(0);
        assert!(outputs.priority_modulation >= 0.0);
    }

    #[test]
    fn test_priority_modulation() {
        let (row_ptr, col_idx) = create_test_graph();
        let initial_conflicts = vec![1.0, 1.0, 1.0];
        let config = DendriticReservoirConfig {
            reservoir_influence: 0.5,
            ..Default::default()
        };

        let reservoir =
            DendriticReservoirCPU::new(3, &row_ptr, &col_idx, &initial_conflicts, config);

        let wavelet_priorities = vec![1.0, 2.0, -1.0];
        let modulated = reservoir.modulate_priorities(&wavelet_priorities);

        assert_eq!(modulated.len(), 3);
        assert_eq!(modulated[2], -1.0); // Non-conflicting unchanged
    }
}
