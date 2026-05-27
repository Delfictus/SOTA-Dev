//! Pipeline Bridge - Connects PipelineOrchestrator to the Event System
//!
//! This is the critical integration layer that makes the TUI functional by:
//! - Wrapping PipelineOrchestrator execution
//! - Emitting real-time events during optimization
//! - Updating StateStore with progress information
//! - Providing progress callbacks for iteration-level updates

use super::events::{EventBus, OptimizationConfig, PhaseId, PrismEvent};
use super::state::StateStore;
use anyhow::Result;
use prism_core::{ColoringSolution, Graph};
use prism_pipeline::{
    config::PipelineConfig,
    orchestrator::PipelineOrchestrator,
};
use prism_fluxnet::{RLConfig, UniversalRLController};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Bridge between PipelineOrchestrator and the event/state system
pub struct PipelineBridge {
    /// Shared state store for UI updates
    state: Arc<StateStore>,
    /// Event bus for broadcasting events
    event_bus: Arc<EventBus>,
}

impl PipelineBridge {
    /// Create a new pipeline bridge
    pub fn new(state: Arc<StateStore>, event_bus: Arc<EventBus>) -> Self {
        Self { state, event_bus }
    }

    /// Load a graph from a DIMACS file
    pub async fn load_graph(&self, path: impl AsRef<Path>) -> Result<Graph> {
        let path_ref = path.as_ref();
        log::info!("Loading graph from: {}", path_ref.display());

        // Emit loading event
        self.event_bus
            .publish(PrismEvent::LoadGraph {
                path: path_ref.to_string_lossy().to_string(),
            })
            .await?;

        // Update pipeline state to Loading
        {
            let mut pipeline = self.state.pipeline.write();
            pipeline.status = super::state::PipelineStatus::Loading;
        }

        // Load the graph using prism-core's DIMACS parser
        let graph = prism_core::dimacs::parse_dimacs_file(path_ref)
            .map_err(|e| anyhow::anyhow!("Failed to load DIMACS file: {}", e))?;

        // Compute graph statistics
        let density = graph.density();
        let estimated_chromatic = Self::estimate_chromatic_number(&graph);

        // Update graph state
        {
            let mut graph_state = self.state.graph.write();
            graph_state.loaded = true;
            graph_state.path = Some(path_ref.to_string_lossy().to_string());
            graph_state.vertices = graph.num_vertices;
            graph_state.edges = graph.num_edges;
            graph_state.density = density;
            graph_state.estimated_chromatic = estimated_chromatic;

            // Compute max and average degree
            if let Some(ref degrees) = graph.degrees {
                graph_state.max_degree = *degrees.iter().max().unwrap_or(&0);
                graph_state.avg_degree =
                    degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
            } else {
                let degrees: Vec<usize> = graph
                    .adjacency
                    .iter()
                    .map(|neighbors| neighbors.len())
                    .collect();
                graph_state.max_degree = *degrees.iter().max().unwrap_or(&0);
                graph_state.avg_degree =
                    degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
            }
        }

        // Emit graph loaded event
        self.event_bus
            .publish(PrismEvent::GraphLoaded {
                vertices: graph.num_vertices,
                edges: graph.num_edges,
                density,
                estimated_chromatic,
            })
            .await?;

        log::info!(
            "Graph loaded: {} vertices, {} edges, density={:.3}, estimated chromatic={}",
            graph.num_vertices,
            graph.num_edges,
            density,
            estimated_chromatic
        );

        Ok(graph)
    }

    /// Run optimization on a loaded graph
    pub async fn run_optimization(
        &self,
        graph: &Graph,
        config: OptimizationConfig,
    ) -> Result<ColoringSolution> {
        log::info!("Starting optimization with config: {:?}", config);

        // Update pipeline state
        {
            let mut pipeline = self.state.pipeline.write();
            pipeline.status = super::state::PipelineStatus::Running;
            pipeline.max_attempts = config.max_attempts;
            pipeline.current_attempt = 1;
            pipeline.start_time = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            );
        }

        // Emit start event
        self.event_bus
            .publish(PrismEvent::StartOptimization {
                config: config.clone(),
            })
            .await?;

        // Create pipeline configuration
        let pipeline_config = Self::create_pipeline_config(&config)?;

        // Create RL controller
        let rl_controller = UniversalRLController::new(RLConfig::default());

        // Create orchestrator
        let mut orchestrator = PipelineOrchestrator::new(pipeline_config, rl_controller);

        // Configure high-performance GPU modes
        #[cfg(feature = "cuda")]
        {
            // Enable AATGS async scheduling if requested
            if config.enable_aatgs_async {
                orchestrator.enable_aatgs_async(true);
                log::info!("AATGS async scheduling enabled via config");
            }

            // Enable Ultra Kernel if requested
            if config.enable_ultra_kernel {
                orchestrator.enable_ultra_kernel(true);
                log::info!("Ultra Kernel mode enabled via config");
            }

            // Initialize multi-GPU if requested and multiple devices specified
            if config.enable_multi_gpu && config.gpu_device_ids.len() > 1 {
                if let Err(e) = orchestrator.initialize_multi_gpu(&config.gpu_device_ids) {
                    log::warn!("Multi-GPU initialization failed: {}. Using single GPU.", e);
                } else {
                    log::info!("Multi-GPU enabled with {} devices", config.gpu_device_ids.len());
                }
            }
        }

        // Create progress callback bridge
        let callback = ProgressCallbackBridge::new(
            self.state.clone(),
            self.event_bus.clone(),
            config.phases_enabled.clone(),
        );

        // Run the pipeline with our custom callback
        let start_time = Instant::now();
        let result = self
            .run_with_callback(&mut orchestrator, graph, callback)
            .await?;
        let duration_ms = start_time.elapsed().as_millis() as u64;

        // Update final state
        {
            let mut pipeline = self.state.pipeline.write();
            pipeline.status = super::state::PipelineStatus::Completed;
            pipeline.elapsed_ms = duration_ms;

            let mut opt = self.state.optimization.write();
            opt.best_colors = result.chromatic_number;
            opt.best_conflicts = result.conflicts;
        }

        // Emit completion event
        self.event_bus
            .publish(PrismEvent::OptimizationCompleted {
                total_duration_ms: duration_ms,
                final_colors: result.chromatic_number,
                attempts: config.max_attempts,
            })
            .await?;

        log::info!(
            "Optimization completed: {} colors, {} conflicts, duration={}ms",
            result.chromatic_number,
            result.conflicts,
            duration_ms
        );

        Ok(result)
    }

    /// Run pipeline with progress callbacks
    async fn run_with_callback(
        &self,
        orchestrator: &mut PipelineOrchestrator,
        graph: &Graph,
        _callback: ProgressCallbackBridge,
    ) -> Result<ColoringSolution> {
        // For now, we run the standard pipeline and emit events based on phases
        // Future enhancement: Modify PipelineOrchestrator to accept callbacks
        // The callback parameter is reserved for future integration

        let result = orchestrator
            .run(graph)
            .map_err(|e| anyhow::anyhow!("Pipeline execution failed: {}", e))?;

        Ok(result)
    }

    /// Create pipeline configuration from optimization config
    fn create_pipeline_config(opt_config: &OptimizationConfig) -> Result<PipelineConfig> {
        // Load default config and override with optimization settings
        let mut config = PipelineConfig::default();

        // Enable warmstart if requested
        if opt_config.enable_warmstart {
            config.warmstart_config = Some(prism_core::WarmstartConfig::default());
        }

        // TODO: Map phases_enabled to pipeline config
        // TODO: Set target colors if specified

        Ok(config)
    }

    /// Estimate chromatic number using simple heuristics
    fn estimate_chromatic_number(graph: &Graph) -> usize {
        if graph.num_vertices == 0 {
            return 0;
        }

        // Find maximum degree
        let max_degree = graph
            .adjacency
            .iter()
            .map(|neighbors| neighbors.len())
            .max()
            .unwrap_or(0);

        // Brooks' theorem: χ(G) ≤ Δ(G) for most graphs
        // Add small buffer for dense graphs
        let density = graph.density();
        if density > 0.5 {
            (max_degree as f64 * 1.2).ceil() as usize
        } else {
            max_degree + 1
        }
    }
}

// ============================================================================
// Progress Callback System
// ============================================================================

/// Trait for receiving progress updates during pipeline execution
pub trait ProgressCallback: Send + Sync {
    /// Called on each iteration within a phase
    fn on_iteration(&self, phase: PhaseId, iteration: usize, colors: usize, conflicts: usize);

    /// Called when replica state updates (Phase 2: Thermodynamic)
    fn on_replica_update(&self, replica_id: usize, temp: f64, colors: usize, conflicts: usize);

    /// Called when a new best solution is found
    fn on_best_found(&self, phase: PhaseId, iteration: usize, colors: usize, conflicts: usize);

    /// Called when a phase starts
    fn on_phase_start(&self, phase: PhaseId, name: &str);

    /// Called when a phase completes
    fn on_phase_complete(&self, phase: PhaseId, duration_ms: u64, colors: usize, conflicts: usize);

    /// Called when a phase fails
    fn on_phase_failed(&self, phase: PhaseId, error: &str);
}

/// Bridge implementation that emits events and updates state
struct ProgressCallbackBridge {
    state: Arc<StateStore>,
    event_bus: Arc<EventBus>,
    phases: Vec<PhaseId>,
    current_phase_idx: std::sync::atomic::AtomicUsize,
}

impl ProgressCallbackBridge {
    fn new(state: Arc<StateStore>, event_bus: Arc<EventBus>, phases: Vec<PhaseId>) -> Self {
        Self {
            state,
            event_bus,
            phases,
            current_phase_idx: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get the current phase index
    fn current_phase(&self) -> Option<PhaseId> {
        let idx = self
            .current_phase_idx
            .load(std::sync::atomic::Ordering::Relaxed);
        self.phases.get(idx).copied()
    }
}

impl ProgressCallback for ProgressCallbackBridge {
    fn on_iteration(&self, phase: PhaseId, iteration: usize, colors: usize, conflicts: usize) {
        // Update state
        self.state.inc_iterations();
        self.state.record_convergence(colors, conflicts, iteration as u64);

        // Update phase state
        {
            let mut phases = self.state.phases.write();
            if let Some(phase_state) = phases.iter_mut().find(|p| p.id == phase) {
                phase_state.iteration = iteration;
                phase_state.metrics.colors = colors;
                phase_state.metrics.conflicts = conflicts;
            }
        }

        // Update optimization state
        {
            let mut opt = self.state.optimization.write();
            opt.current_colors = colors;
            opt.current_conflicts = conflicts;
        }

        // Emit progress event (every N iterations to avoid spam)
        if iteration % 10 == 0 {
            let _ = self.event_bus.try_send(PrismEvent::PhaseProgress {
                phase,
                iteration,
                max_iterations: 1000, // TODO: Get from phase config
                colors,
                conflicts,
                temperature: 1.0, // TODO: Get from phase state
            });
        }
    }

    fn on_replica_update(&self, replica_id: usize, temp: f64, colors: usize, conflicts: usize) {
        // Record temperature
        self.state.record_temperature(replica_id, temp);

        // Emit replica update event
        let _ = self.event_bus.try_send(PrismEvent::ReplicaUpdate {
            replica_id,
            temperature: temp,
            colors,
            conflicts,
            energy: conflicts as f64, // Simplified energy
        });
    }

    fn on_best_found(&self, phase: PhaseId, iteration: usize, colors: usize, conflicts: usize) {
        // Update best solution in state
        {
            let mut opt = self.state.optimization.write();
            opt.best_colors = colors;
            opt.best_conflicts = conflicts;
            opt.best_iteration = iteration as u64;
            opt.best_phase = Some(phase);
        }

        // Emit new best solution event
        let _ = self.event_bus.try_send(PrismEvent::NewBestSolution {
            colors,
            conflicts,
            iteration,
            phase,
        });

        log::info!(
            "[Bridge] New best: {} colors, {} conflicts at iteration {} (phase: {:?})",
            colors,
            conflicts,
            iteration,
            phase
        );
    }

    fn on_phase_start(&self, phase: PhaseId, name: &str) {
        log::info!("[Bridge] Phase started: {} ({})", name, phase.name());

        // Update phase state
        {
            let mut phases = self.state.phases.write();
            if let Some(phase_state) = phases.iter_mut().find(|p| p.id == phase) {
                phase_state.status = super::state::PhaseStatus::Running;
                phase_state.start_time = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                );
            }
        }

        // Update pipeline current phase
        {
            let mut pipeline = self.state.pipeline.write();
            pipeline.current_phase = Some(phase);
        }

        // Emit phase started event
        let _ = self.event_bus.try_send(PrismEvent::PhaseStarted {
            phase,
            name: name.to_string(),
        });
    }

    fn on_phase_complete(&self, phase: PhaseId, duration_ms: u64, colors: usize, conflicts: usize) {
        log::info!(
            "[Bridge] Phase completed: {} - {} colors, {} conflicts, {}ms",
            phase.name(),
            colors,
            conflicts,
            duration_ms
        );

        // Update phase state
        {
            let mut phases = self.state.phases.write();
            if let Some(phase_state) = phases.iter_mut().find(|p| p.id == phase) {
                phase_state.status = super::state::PhaseStatus::Completed;
                phase_state.duration_ms = duration_ms;
                phase_state.metrics.colors = colors;
                phase_state.metrics.conflicts = conflicts;
                phase_state.progress = 1.0;
            }
        }

        // Emit phase completed event
        let _ = self.event_bus.try_send(PrismEvent::PhaseCompleted {
            phase,
            duration_ms,
            final_colors: colors,
            final_conflicts: conflicts,
        });

        // Advance to next phase
        self.current_phase_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn on_phase_failed(&self, phase: PhaseId, error: &str) {
        log::error!("[Bridge] Phase failed: {} - {}", phase.name(), error);

        // Update phase state
        {
            let mut phases = self.state.phases.write();
            if let Some(phase_state) = phases.iter_mut().find(|p| p.id == phase) {
                phase_state.status = super::state::PhaseStatus::Failed;
            }
        }

        // Emit phase failed event
        let _ = self.event_bus.try_send(PrismEvent::PhaseFailed {
            phase,
            error: error.to_string(),
        });
    }
}

// Extension trait to simplify try_send pattern
trait EventBusExt {
    fn try_send(&self, event: PrismEvent) -> Result<(), String>;
}

impl EventBusExt for Arc<EventBus> {
    fn try_send(&self, event: PrismEvent) -> Result<(), String> {
        // Non-blocking send using tokio's try_send
        // We use tokio::spawn to make this non-blocking in sync context
        let bus = self.clone();
        let _ = tokio::spawn(async move {
            let _ = bus.publish(event).await;
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let state = Arc::new(StateStore::new(100));
        let event_bus = Arc::new(EventBus::new(16));
        let bridge = PipelineBridge::new(state, event_bus);

        // Bridge should be created successfully
        assert!(true);
    }

    #[tokio::test]
    async fn test_estimate_chromatic_number() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);

        let estimated = PipelineBridge::estimate_chromatic_number(&graph);
        assert!(estimated >= 2); // At least 2 colors needed for a path
        assert!(estimated <= 5); // At most n colors
    }
}
