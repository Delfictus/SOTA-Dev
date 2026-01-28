//! Reactive Controller - Bridges Runtime Events to UI State
//!
//! This module provides the reactive connection between the PRISM runtime
//! and the TUI application, handling:
//! - Non-blocking event polling from the runtime event bus
//! - Bidirectional command/event flow (UI ↔ Runtime)
//! - Efficient state synchronization without blocking the UI thread
//! - Type-safe mapping from PrismEvent to App state updates
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐         ┌──────────────────┐         ┌─────────────┐
//! │   Runtime   │ Events  │  Reactive        │ Updates │     App     │
//! │  (actors)   │────────>│  Controller      │────────>│   (TUI)     │
//! │             │         │                  │         │             │
//! │             │<────────│  (command_tx)    │<────────│             │
//! └─────────────┘ Commands└──────────────────┘ Actions └─────────────┘
//! ```

use anyhow::{Result, Context};
use tokio::sync::{broadcast, mpsc};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::app::{App, PhaseStatus, PhaseState, OptimizationState, GpuStatus, ReplicaState};
use crate::runtime::events::{PrismEvent, PhaseId, OptimizationConfig};
use crate::runtime::state::StateStore;

/// Configuration for reactive controller behavior
#[derive(Debug, Clone)]
pub struct ReactiveConfig {
    /// Maximum events to process per poll (prevents UI starvation)
    pub max_events_per_poll: usize,
    /// Timeout for non-blocking event receives (microseconds)
    pub poll_timeout_us: u64,
    /// Channel capacity for command queue
    pub command_queue_capacity: usize,
}

impl Default for ReactiveConfig {
    fn default() -> Self {
        Self {
            max_events_per_poll: 50,
            poll_timeout_us: 100,
            command_queue_capacity: 256,
        }
    }
}

/// Reactive controller bridging runtime and UI
pub struct ReactiveController {
    /// Receives events from runtime actors
    event_rx: broadcast::Receiver<PrismEvent>,

    /// Shared state store (for snapshots and direct queries)
    state: Arc<StateStore>,

    /// Sends commands back to runtime
    command_tx: mpsc::Sender<PrismEvent>,

    /// Receives commands from async tasks
    command_rx: mpsc::Receiver<PrismEvent>,

    /// Configuration
    config: ReactiveConfig,

    /// Statistics for monitoring
    stats: ControllerStats,
}

#[derive(Debug, Default)]
struct ControllerStats {
    events_processed: u64,
    commands_sent: u64,
    last_update: Option<Instant>,
    events_per_second: f64,
}

impl ReactiveController {
    /// Create a new reactive controller
    ///
    /// # Arguments
    /// * `event_rx` - Receiver for runtime events
    /// * `state` - Shared state store
    /// * `command_tx` - Sender for runtime commands
    pub fn new(
        event_rx: broadcast::Receiver<PrismEvent>,
        state: Arc<StateStore>,
        command_tx: mpsc::Sender<PrismEvent>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(ReactiveConfig::default().command_queue_capacity);

        Self {
            event_rx,
            state,
            command_tx,
            command_rx: cmd_rx,
            config: ReactiveConfig::default(),
            stats: ControllerStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        event_rx: broadcast::Receiver<PrismEvent>,
        state: Arc<StateStore>,
        command_tx: mpsc::Sender<PrismEvent>,
        config: ReactiveConfig,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(config.command_queue_capacity);

        Self {
            event_rx,
            state,
            command_tx,
            command_rx: cmd_rx,
            config,
            stats: ControllerStats::default(),
        }
    }

    /// Poll for events and update app state (call this in your render loop)
    ///
    /// This is non-blocking and will process up to `max_events_per_poll` events
    /// before returning control to the UI thread.
    pub fn poll_events(&mut self, app: &mut App) -> Result<()> {
        let mut processed = 0;
        let start = Instant::now();

        // Process incoming events from runtime
        while processed < self.config.max_events_per_poll {
            match self.event_rx.try_recv() {
                Ok(event) => {
                    self.handle_event(event, app)
                        .context("Failed to handle event")?;
                    processed += 1;
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(skipped)) => {
                    log::warn!("UI lagging behind runtime: skipped {} events", skipped);
                    // Continue processing - we'll catch up
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    log::error!("Event bus closed");
                    return Err(anyhow::anyhow!("Event bus closed"));
                }
            }
        }

        // Update statistics
        self.stats.events_processed += processed as u64;
        if let Some(last) = self.stats.last_update {
            let elapsed = start.duration_since(last).as_secs_f64();
            if elapsed > 0.0 {
                self.stats.events_per_second = processed as f64 / elapsed;
            }
        }
        self.stats.last_update = Some(start);

        Ok(())
    }

    /// Handle a single event and update app state
    fn handle_event(&mut self, event: PrismEvent, app: &mut App) -> Result<()> {
        match event {
            // ═══════════════════════════════════════════════════════════════
            // Pipeline Events
            // ═══════════════════════════════════════════════════════════════

            PrismEvent::GraphLoaded { vertices, edges, density, estimated_chromatic } => {
                app.optimization.max_iterations = estimated_chromatic * 1000;
                app.dialogue.add_system_message(&format!(
                    "Graph loaded: {} vertices, {} edges (density: {:.1}%)\n\
                     Estimated chromatic number: {}\n\
                     Ready to start optimization.",
                    vertices, edges, density * 100.0, estimated_chromatic
                ));
            }

            PrismEvent::PhaseStarted { phase, name } => {
                let idx = phase.index();
                if idx < app.phases.len() {
                    app.phases[idx].status = PhaseState::Running;
                    app.phases[idx].progress = 0.0;
                }
                app.dialogue.add_system_message(&format!("Starting {}...", name));
            }

            PrismEvent::PhaseProgress {
                phase,
                iteration,
                max_iterations,
                colors,
                conflicts,
                temperature,
            } => {
                let idx = phase.index();
                if idx < app.phases.len() {
                    let progress = (iteration as f64 / max_iterations as f64) * 100.0;
                    app.phases[idx].progress = progress;
                }

                // Update optimization state
                app.optimization.colors = colors;
                app.optimization.conflicts = conflicts;
                app.optimization.iteration = iteration;
                app.optimization.max_iterations = max_iterations;
                app.optimization.temperature = temperature;

                // Add to convergence history (throttle to every 10 iterations)
                if iteration % 10 == 0 {
                    app.optimization.convergence_history.push((iteration, colors));

                    // Keep history bounded
                    if app.optimization.convergence_history.len() > 1000 {
                        app.optimization.convergence_history.remove(0);
                    }
                }
            }

            PrismEvent::PhaseCompleted { phase, duration_ms, final_colors, final_conflicts } => {
                let idx = phase.index();
                if idx < app.phases.len() {
                    app.phases[idx].status = PhaseState::Completed;
                    app.phases[idx].progress = 100.0;
                    app.phases[idx].time_ms = duration_ms;
                }

                app.dialogue.add_system_message(&format!(
                    "{} completed in {:.2}s: {} colors, {} conflicts",
                    phase.name(),
                    duration_ms as f64 / 1000.0,
                    final_colors,
                    final_conflicts
                ));
            }

            PrismEvent::PhaseFailed { phase, error } => {
                let idx = phase.index();
                if idx < app.phases.len() {
                    app.phases[idx].status = PhaseState::Failed;
                }

                app.dialogue.add_system_message(&format!(
                    "❌ {} failed: {}",
                    phase.name(),
                    error
                ));
            }

            PrismEvent::NewBestSolution { colors, conflicts, iteration, phase } => {
                app.optimization.best_colors = colors;
                app.optimization.best_conflicts = conflicts;

                app.dialogue.add_system_message(&format!(
                    "🎯 New best solution: {} colors, {} conflicts (iteration {}, {})",
                    colors, conflicts, iteration, phase.name()
                ));
            }

            PrismEvent::OptimizationCompleted { total_duration_ms, final_colors, attempts } => {
                app.dialogue.add_system_message(&format!(
                    "✓ Optimization completed in {:.2}s\n\
                     Final result: {} colors ({} attempts)",
                    total_duration_ms as f64 / 1000.0,
                    final_colors,
                    attempts
                ));
            }

            // ═══════════════════════════════════════════════════════════════
            // GPU Events
            // ═══════════════════════════════════════════════════════════════

            PrismEvent::GpuStatus {
                device_id,
                name,
                utilization,
                memory_used,
                memory_total,
                temperature,
                power_watts,
            } => {
                app.gpu.name = name;
                app.gpu.utilization = utilization;
                app.gpu.memory_used = memory_used;
                app.gpu.memory_total = memory_total;
                app.gpu.temperature = temperature;
            }

            PrismEvent::KernelLaunched { name, .. } => {
                if !app.gpu.active_kernels.contains(&name) {
                    app.gpu.active_kernels.push(name);
                }
            }

            PrismEvent::KernelCompleted { name, duration_us } => {
                app.gpu.active_kernels.retain(|k| k != &name);
            }

            // ═══════════════════════════════════════════════════════════════
            // Thermodynamic Events (Phase 2)
            // ═══════════════════════════════════════════════════════════════

            PrismEvent::ReplicaUpdate {
                replica_id,
                temperature,
                colors,
                conflicts,
                energy,
            } => {
                // Update or create replica state
                if let Some(replica) = app.optimization.replicas.iter_mut().find(|r| r.temperature == temperature) {
                    replica.colors = colors;
                } else if app.optimization.replicas.len() < 8 {
                    app.optimization.replicas.push(ReplicaState {
                        temperature,
                        colors,
                        is_best: colors == app.optimization.best_colors,
                    });
                }
            }

            PrismEvent::ReplicaExchange { replica_a, replica_b, accepted } => {
                if accepted {
                    log::debug!("Replica exchange: {} <-> {}", replica_a, replica_b);
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // Quantum Events (Phase 3)
            // ═══════════════════════════════════════════════════════════════

            PrismEvent::QuantumState { coherence, top_amplitudes, tunneling_rate } => {
                app.optimization.quantum_coherence = coherence;
                app.optimization.quantum_amplitudes = top_amplitudes;
            }

            PrismEvent::QuantumMeasurement { measured_colors, pre_collapse_entropy } => {
                app.dialogue.add_system_message(&format!(
                    "Quantum measurement: collapsed to {} colors (entropy: {:.3})",
                    measured_colors,
                    pre_collapse_entropy
                ));
            }

            // ═══════════════════════════════════════════════════════════════
            // Dendritic Events (Phase 0)
            // ═══════════════════════════════════════════════════════════════

            PrismEvent::DendriticUpdate { active_neurons, total_neurons, firing_rate, pattern_detected } => {
                if let Some(pattern) = pattern_detected {
                    app.dialogue.add_system_message(&format!(
                        "Dendritic pattern detected: {} ({}/{} neurons active, {:.1}% firing)",
                        pattern,
                        active_neurons,
                        total_neurons,
                        firing_rate * 100.0
                    ));
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // FluxNet RL Events
            // ═══════════════════════════════════════════════════════════════

            PrismEvent::RlAction { state, action, q_value, epsilon } => {
                log::debug!("RL action: {} (Q={:.3}, ε={:.3})", action, q_value, epsilon);
            }

            PrismEvent::RlReward { reward, cumulative_reward } => {
                log::debug!("RL reward: {:.3} (cumulative: {:.3})", reward, cumulative_reward);
            }

            // ═══════════════════════════════════════════════════════════════
            // LBS Events (Biomolecular Mode)
            // ═══════════════════════════════════════════════════════════════

            PrismEvent::ProteinLoaded { pdb_id, residues, atoms, chains } => {
                // Update protein state
                app.protein.name = pdb_id.clone();
                app.protein.residue_count = residues;
                app.protein.atom_count = atoms;
                app.protein.chain_count = chains;

                app.dialogue.add_system_message(&format!(
                    "Protein loaded: {} ({} residues, {} atoms, {} chains)",
                    pdb_id, residues, atoms, chains
                ));
            }

            PrismEvent::LbsPhaseStarted { phase, name } => {
                // Update LBS progress
                app.protein.lbs_progress.current_phase = Some(name.clone());
                app.protein.lbs_progress.phase_iteration = 0;
                app.protein.lbs_progress.phase_max_iterations = 1000; // Default, will be updated

                app.dialogue.add_system_message(&format!("LBS: Starting {}...", name));
            }

            PrismEvent::LbsPhaseProgress { phase, iteration, max_iterations, pockets_found, best_druggability } => {
                // Update LBS progress in real-time
                app.protein.lbs_progress.phase_iteration = iteration;
                app.protein.lbs_progress.phase_max_iterations = max_iterations;
                app.protein.lbs_progress.pockets_detected = pockets_found;
                app.protein.lbs_progress.best_druggability = best_druggability;

                log::debug!(
                    "LBS {:?}: {}/{} - {} pockets, best druggability: {:.2}",
                    phase, iteration, max_iterations, pockets_found, best_druggability
                );
            }

            PrismEvent::LbsPhaseCompleted { phase, duration_ms, pockets_found } => {
                app.dialogue.add_system_message(&format!(
                    "LBS {:?} completed in {:.2}s: {} pockets found",
                    phase, duration_ms as f64 / 1000.0, pockets_found
                ));
            }

            PrismEvent::PocketDetected { pocket_id, volume, druggability, center, residue_count } => {
                use super::app::PocketInfo;

                // Add pocket to the app state (convert f32 to f64)
                app.protein.pockets.push(PocketInfo {
                    id: pocket_id,
                    volume: volume as f64,
                    depth: 0.0, // Will be updated if available
                    druggability: druggability as f64,
                    center: [center[0] as f64, center[1] as f64, center[2] as f64],
                    residues: vec![], // Will be filled later if available
                    hydrophobicity: 0.0,
                    enclosure: 0.0,
                });

                // Sort by druggability (highest first)
                app.protein.pockets.sort_by(|a, b| {
                    b.druggability.partial_cmp(&a.druggability).unwrap()
                });

                app.dialogue.add_system_message(&format!(
                    "Pocket #{}: volume={:.1}Å³, druggability={:.2}, {} residues at ({:.1}, {:.1}, {:.1})",
                    pocket_id, volume, druggability, residue_count, center[0], center[1], center[2]
                ));
            }

            PrismEvent::LbsPredictionComplete { total_pockets, best_pocket_druggability, total_duration_ms, gpu_accelerated } => {
                // Clear current phase to show completion
                app.protein.lbs_progress.current_phase = None;
                app.protein.lbs_progress.gpu_accelerated = gpu_accelerated;

                app.dialogue.add_system_message(&format!(
                    "LBS prediction complete: {} pockets (best: {:.2} druggability) in {:.2}s {}",
                    total_pockets, best_pocket_druggability, total_duration_ms as f64 / 1000.0,
                    if gpu_accelerated { "[GPU]" } else { "[CPU]" }
                ));
            }

            PrismEvent::GnnInference { num_nodes, num_edges, chromatic_prediction, confidence, gpu_used, latency_ms } => {
                log::debug!(
                    "GNN inference: {} nodes, {} edges -> {} colors (conf: {:.2}) in {}ms {}",
                    num_nodes, num_edges, chromatic_prediction, confidence, latency_ms,
                    if gpu_used { "[GPU]" } else { "[CPU]" }
                );
            }

            PrismEvent::SasaComputed { num_atoms, exposed_area, buried_area, latency_ms } => {
                log::debug!(
                    "SASA computed for {} atoms: exposed={:.1}Å², buried={:.1}Å² in {}ms",
                    num_atoms, exposed_area, buried_area, latency_ms
                );
            }

            // ═══════════════════════════════════════════════════════════════
            // System Events
            // ═══════════════════════════════════════════════════════════════

            PrismEvent::Error { source, message, recoverable } => {
                app.dialogue.add_system_message(&format!(
                    "❌ Error from {}: {}\n{}",
                    source,
                    message,
                    if recoverable { "Attempting recovery..." } else { "Fatal error." }
                ));
            }

            PrismEvent::Shutdown => {
                app.should_quit = true;
            }

            // Ignore command events (they're handled by actors, not UI)
            PrismEvent::LoadGraph { .. }
            | PrismEvent::LoadProtein { .. }
            | PrismEvent::StartOptimization { .. }
            | PrismEvent::PauseOptimization
            | PrismEvent::ResumeOptimization
            | PrismEvent::StopOptimization
            | PrismEvent::SetParameter { .. }
            | PrismEvent::MetricRecorded { .. } => {
                // These are commands or logging events - no UI update needed
            }
        }

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Command API - For sending commands to the runtime
    // ═══════════════════════════════════════════════════════════════════════

    /// Load a graph file (async command)
    pub async fn load_graph(&self, path: String) -> Result<()> {
        self.command_tx
            .send(PrismEvent::LoadGraph { path })
            .await
            .context("Failed to send LoadGraph command")?;
        self.update_stats_commands();
        Ok(())
    }

    /// Load a protein structure (async command)
    pub async fn load_protein(&self, path: String) -> Result<()> {
        self.command_tx
            .send(PrismEvent::LoadProtein { path })
            .await
            .context("Failed to send LoadProtein command")?;
        self.update_stats_commands();
        Ok(())
    }

    /// Start optimization with given configuration
    pub async fn start_optimization(&self, config: OptimizationConfig) -> Result<()> {
        self.command_tx
            .send(PrismEvent::StartOptimization { config })
            .await
            .context("Failed to send StartOptimization command")?;
        self.update_stats_commands();
        Ok(())
    }

    /// Pause current optimization
    pub async fn pause(&self) -> Result<()> {
        self.command_tx
            .send(PrismEvent::PauseOptimization)
            .await
            .context("Failed to send PauseOptimization command")?;
        self.update_stats_commands();
        Ok(())
    }

    /// Resume paused optimization
    pub async fn resume(&self) -> Result<()> {
        self.command_tx
            .send(PrismEvent::ResumeOptimization)
            .await
            .context("Failed to send ResumeOptimization command")?;
        self.update_stats_commands();
        Ok(())
    }

    /// Stop optimization
    pub async fn stop(&self) -> Result<()> {
        self.command_tx
            .send(PrismEvent::StopOptimization)
            .await
            .context("Failed to send StopOptimization command")?;
        self.update_stats_commands();
        Ok(())
    }

    /// Set a parameter
    pub async fn set_parameter(&self, key: String, value: crate::runtime::events::ParameterValue) -> Result<()> {
        self.command_tx
            .send(PrismEvent::SetParameter { key, value })
            .await
            .context("Failed to send SetParameter command")?;
        self.update_stats_commands();
        Ok(())
    }

    /// Request runtime shutdown
    pub async fn shutdown(&self) -> Result<()> {
        self.command_tx
            .send(PrismEvent::Shutdown)
            .await
            .context("Failed to send Shutdown command")?;
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // State Access - Direct queries to StateStore
    // ═══════════════════════════════════════════════════════════════════════

    /// Get the current convergence history from state store
    pub fn get_convergence_history(&self) -> Vec<(u64, usize, usize)> {
        use crate::runtime::state::ConvergencePoint;
        self.state.convergence_history.to_vec()
            .into_iter()
            .map(|pt: ConvergencePoint| (pt.iteration, pt.colors, pt.conflicts))
            .collect()
    }

    /// Get GPU utilization history
    pub fn get_gpu_utilization_history(&self) -> Vec<(u64, f64)> {
        use crate::runtime::state::GpuUtilPoint;
        self.state.gpu_utilization_history.to_vec()
            .into_iter()
            .map(|pt: GpuUtilPoint| (pt.timestamp_ms, pt.utilization))
            .collect()
    }

    /// Get temperature history (for thermodynamic visualization)
    pub fn get_temperature_history(&self) -> Vec<(u64, usize, f64)> {
        use crate::runtime::state::TemperaturePoint;
        self.state.temperature_history.to_vec()
            .into_iter()
            .map(|pt: TemperaturePoint| (pt.timestamp_ms, pt.replica_id, pt.temperature))
            .collect()
    }

    /// Get controller statistics
    pub fn stats(&self) -> &ControllerStats {
        &self.stats
    }

    /// Get event processing rate (events/second)
    pub fn event_rate(&self) -> f64 {
        self.stats.events_per_second
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Internal Helpers
    // ═══════════════════════════════════════════════════════════════════════

    fn update_stats_commands(&self) {
        // This is a const fn in practice but we keep it mutable for future stats
        // self.stats.commands_sent += 1; // Would require &mut self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience Builders
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for ReactiveController with fluent API
pub struct ReactiveControllerBuilder {
    config: ReactiveConfig,
}

impl ReactiveControllerBuilder {
    pub fn new() -> Self {
        Self {
            config: ReactiveConfig::default(),
        }
    }

    pub fn max_events_per_poll(mut self, max: usize) -> Self {
        self.config.max_events_per_poll = max;
        self
    }

    pub fn poll_timeout_us(mut self, timeout: u64) -> Self {
        self.config.poll_timeout_us = timeout;
        self
    }

    pub fn command_queue_capacity(mut self, capacity: usize) -> Self {
        self.config.command_queue_capacity = capacity;
        self
    }

    pub fn build(
        self,
        event_rx: broadcast::Receiver<PrismEvent>,
        state: Arc<StateStore>,
        command_tx: mpsc::Sender<PrismEvent>,
    ) -> ReactiveController {
        ReactiveController::with_config(event_rx, state, command_tx, self.config)
    }
}

impl Default for ReactiveControllerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::events::EventBus;

    #[tokio::test]
    async fn test_reactive_controller_event_handling() {
        let event_bus = EventBus::new(16);
        let state = Arc::new(StateStore::new(100));
        let (cmd_tx, mut cmd_rx) = mpsc::channel(16);

        let event_rx = event_bus.subscribe();
        let mut controller = ReactiveController::new(event_rx, state.clone(), cmd_tx);

        // Create a test app
        let mut app = App::new(None, "coloring".into(), 0).unwrap();

        // Publish a test event
        event_bus.publish(PrismEvent::GraphLoaded {
            vertices: 500,
            edges: 12500,
            density: 0.1,
            estimated_chromatic: 48,
        }).await.unwrap();

        // Small delay to ensure event is processed
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Poll events
        controller.poll_events(&mut app).unwrap();

        // Verify app state updated
        assert_eq!(app.optimization.max_iterations, 48000);
    }

    #[tokio::test]
    async fn test_reactive_controller_commands() {
        let event_bus = EventBus::new(16);
        let state = Arc::new(StateStore::new(100));
        let (cmd_tx, mut cmd_rx) = mpsc::channel(16);

        let event_rx = event_bus.subscribe();
        let controller = ReactiveController::new(event_rx, state.clone(), cmd_tx);

        // Send a command
        controller.load_graph("/tmp/test.col".into()).await.unwrap();

        // Verify command received
        let cmd = cmd_rx.recv().await.unwrap();
        match cmd {
            PrismEvent::LoadGraph { path } => assert_eq!(path, "/tmp/test.col"),
            _ => panic!("Wrong command type"),
        }
    }
}
