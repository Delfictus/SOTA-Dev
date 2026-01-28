//! PRISM Application State
//!
//! Core application state and event loop for the TUI.

use anyhow::Result;
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{prelude::*, widgets::*};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

use super::render;
use super::theme::Theme;
use crate::ai::AiDialogue;
use crate::runtime::events::PrismEvent;
use crate::runtime::state::StateStore;
use crate::streaming::PipelineStream;
use crate::widgets;

/// Application mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// Graph coloring mode
    GraphColoring,
    /// Biomolecular/protein mode
    Biomolecular,
    /// Materials science mode
    Materials,
    /// Welcome/selection screen
    Welcome,
}

/// Current view/panel focus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Main,
    Dialogue,
    Metrics,
    Pipeline,
    Graph,
    Protein,
}

/// Phase execution status
#[derive(Debug, Clone)]
pub struct PhaseStatus {
    pub name: String,
    pub progress: f64,
    pub status: PhaseState,
    pub time_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseState {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Optimization state
#[derive(Debug, Clone, Default)]
pub struct OptimizationState {
    pub colors: usize,
    pub conflicts: usize,
    pub best_colors: usize,
    pub best_conflicts: usize,
    pub iteration: usize,
    pub max_iterations: usize,
    pub temperature: f64,
    pub convergence_history: Vec<(usize, usize)>, // (iteration, colors)
    pub replicas: Vec<ReplicaState>,
    pub quantum_amplitudes: Vec<(usize, f64)>, // (color, amplitude)
    pub quantum_coherence: f64,
    pub dendritic_active_neurons: usize,
    pub dendritic_total_neurons: usize,
    pub dendritic_firing_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ReplicaState {
    pub temperature: f64,
    pub colors: usize,
    pub is_best: bool,
}

/// GPU status
#[derive(Debug, Clone, Default)]
pub struct GpuStatus {
    pub name: String,
    pub utilization: f64,
    pub memory_used: u64,
    pub memory_total: u64,
    pub temperature: u32,
    pub active_kernels: Vec<String>,
}

/// Protein analysis state
#[derive(Debug, Clone, Default)]
pub struct ProteinState {
    pub name: String,
    pub residue_count: usize,
    pub atom_count: usize,
    pub chain_count: usize,
    pub pockets: Vec<PocketInfo>,
    pub gnn_attention: Vec<(String, f64)>, // (residue, attention)
    pub binding_affinity: f64,
    pub druggability: f64,
    pub lbs_progress: LbsProgress,
}

#[derive(Debug, Clone)]
pub struct PocketInfo {
    pub id: usize,
    pub volume: f64,
    pub depth: f64,
    pub druggability: f64,
    pub center: [f64; 3],
    pub residues: Vec<String>,
    pub hydrophobicity: f64,
    pub enclosure: f64,
}

/// LBS pipeline progress tracking
#[derive(Debug, Clone, Default)]
pub struct LbsProgress {
    pub current_phase: Option<String>,
    pub pockets_detected: usize,
    pub best_druggability: f32,
    pub phase_iteration: usize,
    pub phase_max_iterations: usize,
    pub gpu_accelerated: bool,
}

/// Main application state
pub struct App {
    /// Current mode
    pub mode: AppMode,

    /// Current focus
    pub focus: Focus,

    /// Should the app quit?
    pub should_quit: bool,

    /// Input file path
    pub input_path: Option<String>,

    /// AI dialogue system
    pub dialogue: AiDialogue,

    /// Current user input
    pub input_buffer: String,

    /// Optimization state
    pub optimization: OptimizationState,

    /// GPU status
    pub gpu: GpuStatus,

    /// Protein state (for biomolecular mode)
    pub protein: ProteinState,

    /// Phase statuses
    pub phases: Vec<PhaseStatus>,

    /// Pipeline stream (for real-time updates)
    pipeline_stream: Option<PipelineStream>,

    /// Last update time
    last_update: Instant,

    /// Frame count for animations
    frame: u64,

    /// Show help overlay
    pub show_help: bool,

    /// GPU device ID
    gpu_device: usize,

    /// Runtime state store (shared with runtime actors)
    runtime_state: Option<Arc<StateStore>>,

    /// Event receiver for runtime updates
    event_receiver: Option<broadcast::Receiver<PrismEvent>>,
}

impl App {
    /// Create a new application instance
    pub fn new(input: Option<String>, mode: String, gpu_device: usize) -> Result<Self> {
        let app_mode = match mode.as_str() {
            "coloring" => AppMode::GraphColoring,
            "biomolecular" | "bio" | "protein" => AppMode::Biomolecular,
            "materials" | "mat" => AppMode::Materials,
            _ => {
                // Auto-detect based on input file extension
                if let Some(ref path) = input {
                    if path.ends_with(".col") || path.ends_with(".clq") {
                        AppMode::GraphColoring
                    } else if path.ends_with(".pdb") || path.ends_with(".fasta") {
                        AppMode::Biomolecular
                    } else {
                        AppMode::Welcome
                    }
                } else {
                    AppMode::Welcome
                }
            }
        };

        let phases = vec![
            PhaseStatus {
                name: "P0-Dendritic".into(),
                progress: 0.0,
                status: PhaseState::Pending,
                time_ms: 0,
            },
            PhaseStatus {
                name: "P1-Inference".into(),
                progress: 0.0,
                status: PhaseState::Pending,
                time_ms: 0,
            },
            PhaseStatus {
                name: "P2-Thermo".into(),
                progress: 0.0,
                status: PhaseState::Pending,
                time_ms: 0,
            },
            PhaseStatus {
                name: "P3-Quantum".into(),
                progress: 0.0,
                status: PhaseState::Pending,
                time_ms: 0,
            },
            PhaseStatus {
                name: "P4-Geodesic".into(),
                progress: 0.0,
                status: PhaseState::Pending,
                time_ms: 0,
            },
            PhaseStatus {
                name: "P6-TDA".into(),
                progress: 0.0,
                status: PhaseState::Pending,
                time_ms: 0,
            },
            PhaseStatus {
                name: "P7-Ensemble".into(),
                progress: 0.0,
                status: PhaseState::Pending,
                time_ms: 0,
            },
        ];

        Ok(Self {
            mode: app_mode,
            focus: Focus::Dialogue,
            should_quit: false,
            input_path: input,
            dialogue: AiDialogue::new(),
            input_buffer: String::new(),
            optimization: OptimizationState::default(),
            gpu: GpuStatus {
                name: "RTX 3060".into(),
                utilization: 0.0,
                memory_used: 0,
                memory_total: 12 * 1024 * 1024 * 1024,
                temperature: 45,
                active_kernels: vec![],
            },
            protein: ProteinState::default(),
            phases,
            pipeline_stream: None,
            last_update: Instant::now(),
            frame: 0,
            show_help: false,
            gpu_device,
            runtime_state: None,
            event_receiver: None,
        })
    }

    /// Create a new application instance with runtime integration
    pub fn new_with_runtime(
        input: Option<String>,
        mode: String,
        gpu_device: usize,
        runtime_state: Arc<StateStore>,
        event_receiver: broadcast::Receiver<PrismEvent>,
    ) -> Result<Self> {
        let mut app = Self::new(input, mode, gpu_device)?;
        app.runtime_state = Some(runtime_state);
        app.event_receiver = Some(event_receiver);
        Ok(app)
    }

    /// Main event loop
    pub fn run(&mut self, terminal: &mut Terminal<impl Backend>) -> Result<()> {
        // Show welcome message
        self.dialogue.add_system_message(&format!(
            "Welcome to PRISM-Fold v{}.\n\n\
             I'm your computational research partner. I can help you with:\n\
             • Graph coloring optimization\n\
             • Protein binding site prediction\n\
             • Materials property analysis\n\n\
             {}",
            env!("CARGO_PKG_VERSION"),
            if let Some(ref path) = self.input_path {
                format!("I see you've loaded: {}\n\nReady to analyze when you are.", path)
            } else {
                "Load a file to get started, or ask me anything.".into()
            }
        ));

        loop {
            // Render
            terminal.draw(|frame| self.render(frame))?;

            // Handle events with timeout for animations
            if event::poll(Duration::from_millis(50))? {
                if let CrosstermEvent::Key(key) = event::read()? {
                    self.handle_key(key)?;
                }
            }

            // Update state (animations, streaming data)
            self.update()?;

            if self.should_quit {
                break;
            }
        }

        Ok(())
    }

    /// Handle keyboard input
    fn handle_key(&mut self, key: KeyEvent) -> Result<()> {
        // Global shortcuts
        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c')) => {
                self.should_quit = true;
                return Ok(());
            }
            (KeyModifiers::CONTROL, KeyCode::Char('q')) => {
                self.should_quit = true;
                return Ok(());
            }
            (_, KeyCode::F(1)) => {
                self.show_help = !self.show_help;
                return Ok(());
            }
            (_, KeyCode::Esc) => {
                if self.show_help {
                    self.show_help = false;
                    return Ok(());
                }
            }
            _ => {}
        }

        // Mode-specific input handling
        match self.focus {
            Focus::Dialogue => self.handle_dialogue_input(key)?,
            _ => self.handle_navigation(key)?,
        }

        Ok(())
    }

    /// Handle dialogue input
    fn handle_dialogue_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Enter => {
                if !self.input_buffer.is_empty() {
                    let input = std::mem::take(&mut self.input_buffer);
                    self.process_command(&input)?;
                }
            }
            KeyCode::Char(c) => {
                self.input_buffer.push(c);
            }
            KeyCode::Backspace => {
                self.input_buffer.pop();
            }
            KeyCode::Tab => {
                // Cycle focus
                self.focus = match self.focus {
                    Focus::Dialogue => Focus::Main,
                    Focus::Main => Focus::Metrics,
                    Focus::Metrics => Focus::Pipeline,
                    Focus::Pipeline => Focus::Dialogue,
                    _ => Focus::Dialogue,
                };
            }
            _ => {}
        }
        Ok(())
    }

    /// Handle navigation keys
    fn handle_navigation(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Tab => {
                self.focus = Focus::Dialogue;
            }
            KeyCode::Char('g') => {
                self.focus = Focus::Graph;
            }
            KeyCode::Char('p') => {
                self.focus = Focus::Protein;
            }
            KeyCode::Char('m') => {
                self.focus = Focus::Metrics;
            }
            _ => {}
        }
        Ok(())
    }

    /// Process user command
    fn process_command(&mut self, input: &str) -> Result<()> {
        // Add user message to dialogue
        self.dialogue.add_user_message(input);

        // Parse and execute command
        let response = self.execute_command(input)?;

        // Add AI response
        self.dialogue.add_system_message(&response);

        Ok(())
    }

    /// Execute a command and return response
    fn execute_command(&mut self, input: &str) -> Result<String> {
        let input_lower = input.to_lowercase();

        // Handle special commands
        if input_lower.starts_with("load ") {
            let path = input[5..].trim();
            return self.load_file(path);
        }

        if input_lower == "run" || input_lower == "go" || input_lower == "solve" {
            return match self.mode {
                AppMode::Biomolecular => self.start_lbs_prediction(),
                _ => self.start_optimization(),
            };
        }

        if input_lower == "analyze" || input_lower == "predict" {
            if self.mode == AppMode::Biomolecular {
                return self.start_lbs_prediction();
            }
        }

        if input_lower == "stop" || input_lower == "pause" {
            return Ok("Optimization paused. Type 'resume' to continue.".into());
        }

        if input_lower == "status" {
            return Ok(self.get_status_report());
        }

        if input_lower.starts_with("set ") {
            return self.handle_set_command(&input[4..]);
        }

        // Default: AI-style conversational response
        Ok(self.generate_ai_response(input))
    }

    /// Load a file
    fn load_file(&mut self, path: &str) -> Result<String> {
        self.input_path = Some(path.to_string());

        // Auto-detect mode
        if path.ends_with(".col") || path.ends_with(".clq") {
            self.mode = AppMode::GraphColoring;
            Ok(format!(
                "Loaded graph file: {}\n\n\
                 Detected DIMACS format. Ready for graph coloring optimization.\n\
                 Type 'run' to start, or ask me about the graph.",
                path
            ))
        } else if path.ends_with(".pdb") {
            self.mode = AppMode::Biomolecular;
            Ok(format!(
                "Loaded protein structure: {}\n\n\
                 Ready for binding site analysis.\n\
                 Type 'run' to start pocket detection, or ask me about the structure.",
                path
            ))
        } else {
            Ok(format!("Loaded: {}\n\nFile type not recognized. Please specify mode.", path))
        }
    }

    /// Start optimization
    fn start_optimization(&mut self) -> Result<String> {
        if self.input_path.is_none() {
            return Ok("No file loaded. Use 'load <path>' first.".into());
        }

        // Initialize phases
        for phase in &mut self.phases {
            phase.status = PhaseState::Pending;
            phase.progress = 0.0;
        }

        // Start first phase
        self.phases[0].status = PhaseState::Running;

        Ok(format!(
            "Starting optimization on {}...\n\n\
             I'll keep you updated on progress. Feel free to ask questions or adjust parameters.",
            self.input_path.as_ref().unwrap()
        ))
    }

    /// Start LBS (Ligand Binding Site) prediction pipeline
    fn start_lbs_prediction(&mut self) -> Result<String> {
        if self.input_path.is_none() {
            return Ok("No PDB file loaded. Use 'load <path.pdb>' first.".into());
        }

        // Initialize LBS-specific phases
        for phase in &mut self.phases {
            phase.status = PhaseState::Pending;
            phase.progress = 0.0;
        }

        // Clear previous pockets
        self.protein.pockets.clear();

        // Start first phase
        self.phases[0].status = PhaseState::Running;

        Ok(format!(
            "Starting LBS prediction on {}...\n\n\
             Running 7-phase pocket detection with GPU acceleration.\n\
             I'll report binding sites as they're detected.",
            self.input_path.as_ref().unwrap()
        ))
    }

    /// Handle set command
    fn handle_set_command(&mut self, args: &str) -> Result<String> {
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok("Usage: set <parameter> <value>".into());
        }

        let param = parts[0];
        let value = parts[1];

        match param {
            "iterations" | "iter" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.optimization.max_iterations = v;
                    Ok(format!("Set max iterations to {}", v))
                } else {
                    Ok("Invalid value for iterations".into())
                }
            }
            "temperature" | "temp" => {
                if let Ok(v) = value.parse::<f64>() {
                    self.optimization.temperature = v;
                    Ok(format!("Set temperature to {:.3}", v))
                } else {
                    Ok("Invalid value for temperature".into())
                }
            }
            _ => Ok(format!("Unknown parameter: {}", param)),
        }
    }

    /// Get status report
    fn get_status_report(&self) -> String {
        format!(
            "Current Status:\n\
             ═══════════════════════════════════\n\
             Mode: {:?}\n\
             Colors: {} (best: {})\n\
             Conflicts: {} (best: {})\n\
             Iteration: {}/{}\n\
             GPU: {} ({:.0}%, {}°C)\n\
             ═══════════════════════════════════",
            self.mode,
            self.optimization.colors,
            self.optimization.best_colors,
            self.optimization.conflicts,
            self.optimization.best_conflicts,
            self.optimization.iteration,
            self.optimization.max_iterations,
            self.gpu.name,
            self.gpu.utilization,
            self.gpu.temperature
        )
    }

    /// Generate AI-style response
    fn generate_ai_response(&self, input: &str) -> String {
        // Simple pattern matching for demo - would be more sophisticated in production
        let input_lower = input.to_lowercase();

        if input_lower.contains("help") {
            return "I can help with:\n\
                    • load <file> - Load a graph or protein\n\
                    • run/solve - Start optimization\n\
                    • set <param> <value> - Adjust parameters\n\
                    • status - Show current status\n\
                    • explain <topic> - Explain what's happening\n\n\
                    Or just ask me anything about the optimization!".into();
        }

        if input_lower.contains("explain") {
            if input_lower.contains("phase 2") || input_lower.contains("thermo") {
                return "Phase 2 (Thermodynamic) uses parallel tempering - a powerful optimization technique.\n\n\
                        Multiple 'replicas' explore the solution space at different temperatures:\n\
                        • High temperature: Explores broadly, can escape local minima\n\
                        • Low temperature: Refines solutions, finds precise optima\n\n\
                        Replicas periodically exchange solutions, combining exploration and exploitation.".into();
            }
            if input_lower.contains("quantum") || input_lower.contains("phase 3") {
                return "Phase 3 (Quantum) uses quantum-inspired amplitude evolution.\n\n\
                        Each vertex has a superposition of color probabilities. The quantum kernel:\n\
                        • Evolves amplitudes based on neighbor constraints\n\
                        • Uses tunneling to pass through energy barriers\n\
                        • Collapses to a classical solution via measurement\n\n\
                        This helps escape local minima that classical methods get stuck in.".into();
            }
        }

        if input_lower.contains("how") && input_lower.contains("long") {
            return format!(
                "Based on current progress ({}/{} iterations), \
                 I estimate completion in approximately {} seconds.",
                self.optimization.iteration,
                self.optimization.max_iterations,
                (self.optimization.max_iterations - self.optimization.iteration) / 100
            );
        }

        // Default response
        "I understand. Let me know if you'd like me to explain what's happening, \
         adjust parameters, or help with anything else.".into()
    }

    /// Update state (called each frame)
    fn update(&mut self) -> Result<()> {
        self.frame += 1;

        // Poll runtime events if connected to runtime
        if self.event_receiver.is_some() {
            // Process all available events without blocking
            loop {
                let event_result = self
                    .event_receiver
                    .as_mut()
                    .unwrap()
                    .try_recv();

                match event_result {
                    Ok(event) => {
                        self.process_runtime_event(event)?;
                    }
                    Err(broadcast::error::TryRecvError::Empty) => {
                        // No more events available
                        break;
                    }
                    Err(broadcast::error::TryRecvError::Lagged(skipped)) => {
                        log::warn!("UI lagged behind runtime, skipped {} events", skipped);
                        // Continue processing newer events
                    }
                    Err(broadcast::error::TryRecvError::Closed) => {
                        log::error!("Runtime event channel closed");
                        break;
                    }
                }
            }
        } else {
            // Fallback: simulate progress for demo (when not connected to runtime)
            if self.frame % 20 == 0 {
                // Update running phase
                for phase in &mut self.phases {
                    if phase.status == PhaseState::Running {
                        phase.progress += 0.5;
                        if phase.progress >= 100.0 {
                            phase.status = PhaseState::Completed;
                            phase.progress = 100.0;
                        }
                        break;
                    }
                }

                // Start next phase if previous completed
                let mut start_next = false;
                for i in 0..self.phases.len() {
                    if self.phases[i].status == PhaseState::Completed && i + 1 < self.phases.len() {
                        if self.phases[i + 1].status == PhaseState::Pending {
                            start_next = true;
                        }
                    }
                }
                if start_next {
                    for phase in &mut self.phases {
                        if phase.status == PhaseState::Pending {
                            phase.status = PhaseState::Running;
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Process a runtime event and update UI state
    fn process_runtime_event(&mut self, event: PrismEvent) -> Result<()> {
        use crate::runtime::events::PhaseId;

        match event {
            PrismEvent::GraphLoaded { vertices, edges, density, estimated_chromatic } => {
                self.dialogue.add_system_message(&format!(
                    "Graph loaded: {} vertices, {} edges (density: {:.2}%)\n\
                     Estimated chromatic number: {}",
                    vertices, edges, density * 100.0, estimated_chromatic
                ));
            }

            PrismEvent::PhaseStarted { phase, name } => {
                let phase_idx = phase.index();
                if phase_idx < self.phases.len() {
                    self.phases[phase_idx].status = PhaseState::Running;
                    self.phases[phase_idx].progress = 0.0;
                }
            }

            PrismEvent::PhaseProgress { phase, iteration, max_iterations, colors, conflicts, temperature } => {
                let phase_idx = phase.index();
                if phase_idx < self.phases.len() {
                    self.phases[phase_idx].progress = (iteration as f64 / max_iterations as f64) * 100.0;
                }

                // Update optimization state
                self.optimization.colors = colors;
                self.optimization.conflicts = conflicts;
                self.optimization.iteration = iteration;
                self.optimization.max_iterations = max_iterations;
                self.optimization.temperature = temperature;
            }

            PrismEvent::PhaseCompleted { phase, duration_ms, final_colors, final_conflicts } => {
                let phase_idx = phase.index();
                if phase_idx < self.phases.len() {
                    self.phases[phase_idx].status = PhaseState::Completed;
                    self.phases[phase_idx].progress = 100.0;
                    self.phases[phase_idx].time_ms = duration_ms;
                }
            }

            PrismEvent::PhaseFailed { phase, error } => {
                let phase_idx = phase.index();
                if phase_idx < self.phases.len() {
                    self.phases[phase_idx].status = PhaseState::Failed;
                }
                self.dialogue.add_system_message(&format!("Phase {} failed: {}", phase.name(), error));
            }

            PrismEvent::NewBestSolution { colors, conflicts, iteration, phase } => {
                self.optimization.best_colors = colors;
                self.optimization.best_conflicts = conflicts;

                self.dialogue.add_system_message(&format!(
                    "New best solution! {} colors, {} conflicts (iteration {}, {})",
                    colors, conflicts, iteration, phase.name()
                ));
            }

            PrismEvent::GpuStatus { device_id, name, utilization, memory_used, memory_total, temperature, power_watts } => {
                if device_id == self.gpu_device {
                    self.gpu.name = name;
                    self.gpu.utilization = utilization;
                    self.gpu.memory_used = memory_used;
                    self.gpu.memory_total = memory_total;
                    self.gpu.temperature = temperature;
                }
            }

            PrismEvent::ReplicaUpdate { replica_id, temperature, colors, conflicts, energy } => {
                // Update replica states for visualization
                while self.optimization.replicas.len() <= replica_id {
                    self.optimization.replicas.push(ReplicaState {
                        temperature: 0.0,
                        colors: 0,
                        is_best: false,
                    });
                }

                if replica_id < self.optimization.replicas.len() {
                    self.optimization.replicas[replica_id].temperature = temperature;
                    self.optimization.replicas[replica_id].colors = colors;
                    // Mark best replica
                    self.optimization.replicas[replica_id].is_best =
                        colors <= self.optimization.best_colors;
                }
            }

            PrismEvent::QuantumState { coherence, top_amplitudes, tunneling_rate } => {
                self.optimization.quantum_coherence = coherence;
                self.optimization.quantum_amplitudes = top_amplitudes;
            }

            PrismEvent::DendriticUpdate { active_neurons, total_neurons, firing_rate, pattern_detected } => {
                self.optimization.dendritic_active_neurons = active_neurons;
                self.optimization.dendritic_total_neurons = total_neurons;
                self.optimization.dendritic_firing_rate = firing_rate;

                if let Some(pattern) = pattern_detected {
                    self.dialogue.add_system_message(&format!("Pattern detected: {}", pattern));
                }
            }

            PrismEvent::OptimizationCompleted { total_duration_ms, final_colors, attempts } => {
                self.dialogue.add_system_message(&format!(
                    "Optimization complete! Final result: {} colors in {:.2}s ({} attempts)",
                    final_colors,
                    total_duration_ms as f64 / 1000.0,
                    attempts
                ));
            }

            PrismEvent::Error { source, message, recoverable } => {
                self.dialogue.add_system_message(&format!(
                    "Error in {}: {} ({})",
                    source,
                    message,
                    if recoverable { "recoverable" } else { "fatal" }
                ));
            }

            _ => {
                // Ignore other events
            }
        }

        Ok(())
    }

    /// Render the application
    fn render(&self, frame: &mut Frame) {
        render::render(self, frame);
    }
}
