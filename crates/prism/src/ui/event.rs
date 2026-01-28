//! Event handling for PRISM TUI

use crossterm::event::KeyEvent;

/// Application events
#[derive(Debug, Clone)]
pub enum Event {
    /// Keyboard input
    Key(KeyEvent),

    /// Pipeline update
    PipelineUpdate(PipelineEvent),

    /// GPU status update
    GpuUpdate(GpuEvent),

    /// Tick for animations
    Tick,

    /// Quit signal
    Quit,
}

/// Pipeline events from streaming
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    PhaseStarted { name: String },
    PhaseProgress { name: String, progress: f64 },
    PhaseCompleted { name: String, time_ms: u64 },
    PhaseFailed { name: String, error: String },
    SolutionUpdated { colors: usize, conflicts: usize },
    IterationCompleted { iteration: usize, temperature: f64 },
    ReplicaExchange { from: usize, to: usize },
    QuantumMeasurement { amplitudes: Vec<(usize, f64)> },
    OptimizationComplete { colors: usize, conflicts: usize, time_s: f64 },
}

/// GPU events
#[derive(Debug, Clone)]
pub enum GpuEvent {
    UtilizationUpdate { percent: f64 },
    MemoryUpdate { used: u64, total: u64 },
    TemperatureUpdate { celsius: u32 },
    KernelStarted { name: String },
    KernelCompleted { name: String, time_ms: f64 },
}
