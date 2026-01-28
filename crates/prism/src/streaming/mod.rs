//! Real-time Pipeline Data Streaming
//!
//! Streams optimization data from the pipeline to the TUI.

use tokio::sync::mpsc;

use crate::ui::event::PipelineEvent;

/// Pipeline data stream
pub struct PipelineStream {
    receiver: mpsc::Receiver<PipelineEvent>,
}

impl PipelineStream {
    /// Create a new pipeline stream
    pub fn new() -> (Self, mpsc::Sender<PipelineEvent>) {
        let (sender, receiver) = mpsc::channel(100);
        (Self { receiver }, sender)
    }

    /// Try to receive the next event (non-blocking)
    pub fn try_recv(&mut self) -> Option<PipelineEvent> {
        self.receiver.try_recv().ok()
    }
}

/// Telemetry collector for sending events to the TUI
pub struct TelemetryCollector {
    sender: mpsc::Sender<PipelineEvent>,
}

impl TelemetryCollector {
    pub fn new(sender: mpsc::Sender<PipelineEvent>) -> Self {
        Self { sender }
    }

    /// Send a phase started event
    pub async fn phase_started(&self, name: &str) {
        let _ = self.sender.send(PipelineEvent::PhaseStarted {
            name: name.to_string(),
        }).await;
    }

    /// Send a phase progress event
    pub async fn phase_progress(&self, name: &str, progress: f64) {
        let _ = self.sender.send(PipelineEvent::PhaseProgress {
            name: name.to_string(),
            progress,
        }).await;
    }

    /// Send a phase completed event
    pub async fn phase_completed(&self, name: &str, time_ms: u64) {
        let _ = self.sender.send(PipelineEvent::PhaseCompleted {
            name: name.to_string(),
            time_ms,
        }).await;
    }

    /// Send a solution update
    pub async fn solution_updated(&self, colors: usize, conflicts: usize) {
        let _ = self.sender.send(PipelineEvent::SolutionUpdated {
            colors,
            conflicts,
        }).await;
    }

    /// Send iteration completed
    pub async fn iteration_completed(&self, iteration: usize, temperature: f64) {
        let _ = self.sender.send(PipelineEvent::IterationCompleted {
            iteration,
            temperature,
        }).await;
    }

    /// Send optimization complete
    pub async fn optimization_complete(&self, colors: usize, conflicts: usize, time_s: f64) {
        let _ = self.sender.send(PipelineEvent::OptimizationComplete {
            colors,
            conflicts,
            time_s,
        }).await;
    }
}
