//! PRISM TUI Rendering
//!
//! World-class visualization rendering for the terminal interface.

use ratatui::{
    prelude::*,
    widgets::*,
};

use super::app::{App, AppMode, PhaseState, Focus};
use super::theme::Theme;
use crate::widgets;

/// Main render function
pub fn render(app: &App, frame: &mut Frame) {
    let area = frame.area();

    // Clear background
    frame.render_widget(
        Block::default().style(Style::default().bg(Theme::BG_PRIMARY)),
        area,
    );

    // Main layout: header, content, footer
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(20),    // Content
            Constraint::Length(3),  // Dialogue input
        ])
        .split(area);

    // Render header
    render_header(app, frame, main_layout[0]);

    // Render main content based on mode
    match app.mode {
        AppMode::GraphColoring => render_graph_mode(app, frame, main_layout[1]),
        AppMode::Biomolecular => render_biomolecular_mode(app, frame, main_layout[1]),
        AppMode::Welcome => render_welcome(app, frame, main_layout[1]),
        _ => render_welcome(app, frame, main_layout[1]),
    }

    // Render dialogue input
    render_dialogue_input(app, frame, main_layout[2]);

    // Render help overlay if active
    if app.show_help {
        render_help_overlay(frame, area);
    }
}

/// Render the header bar
fn render_header(app: &App, frame: &mut Frame, area: Rect) {
    let header_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(30),  // Title
            Constraint::Min(20),     // Mode/file
            Constraint::Length(35),  // GPU status
        ])
        .split(area);

    // Title
    let title = Paragraph::new(format!(" ◆ PRISM-Fold v{}", env!("CARGO_PKG_VERSION")))
        .style(Theme::title())
        .block(Block::default().borders(Borders::ALL).border_style(Theme::panel_border()));
    frame.render_widget(title, header_layout[0]);

    // Mode and file
    let mode_text = match app.mode {
        AppMode::GraphColoring => "Graph Coloring",
        AppMode::Biomolecular => "Biomolecular",
        AppMode::Materials => "Materials",
        AppMode::Welcome => "Welcome",
    };
    let file_text = app.input_path.as_deref().unwrap_or("No file loaded");
    let mode = Paragraph::new(format!(" {} │ {}", mode_text, file_text))
        .style(Theme::normal())
        .block(Block::default().borders(Borders::ALL).border_style(Theme::panel_border()));
    frame.render_widget(mode, header_layout[1]);

    // GPU status
    let gpu_color = Theme::gpu_util_color(app.gpu.utilization);
    let gpu_bar = format!(
        "{}",
        "█".repeat((app.gpu.utilization / 10.0) as usize)
    );
    let gpu_text = format!(
        " {} {} {:.0}% {}°C",
        app.gpu.name,
        gpu_bar,
        app.gpu.utilization,
        app.gpu.temperature
    );
    let gpu = Paragraph::new(gpu_text)
        .style(Style::default().fg(gpu_color))
        .block(Block::default().borders(Borders::ALL).border_style(Theme::panel_border()));
    frame.render_widget(gpu, header_layout[2]);
}

/// Render graph coloring mode
fn render_graph_mode(app: &App, frame: &mut Frame, area: Rect) {
    // Split into main content and dialogue
    let content_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(65),  // Visualizations
            Constraint::Percentage(35),  // Dialogue + metrics
        ])
        .split(area);

    // Left side: visualizations
    let viz_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),  // Graph + energy landscape
            Constraint::Percentage(25),  // Replica swarm
            Constraint::Percentage(25),  // Quantum + dendritic + kernels
        ])
        .split(content_layout[0]);

    // Top row: graph and energy landscape
    let top_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(viz_layout[0]);

    render_graph_visualization(app, frame, top_row[0]);
    render_energy_landscape(app, frame, top_row[1]);

    // Middle: replica swarm
    render_replica_swarm(app, frame, viz_layout[1]);

    // Bottom row: quantum, dendritic, kernels
    let bottom_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(viz_layout[2]);

    render_quantum_state(app, frame, bottom_row[0]);
    render_dendritic_activity(app, frame, bottom_row[1]);
    render_gpu_kernels(app, frame, bottom_row[2]);

    // Right side: dialogue and metrics
    let right_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),     // Dialogue
            Constraint::Length(10),  // Pipeline
            Constraint::Length(8),   // Convergence
        ])
        .split(content_layout[1]);

    render_dialogue_history(app, frame, right_layout[0]);
    render_pipeline_flow(app, frame, right_layout[1]);
    render_convergence_chart(app, frame, right_layout[2]);
}

/// Render biomolecular mode
fn render_biomolecular_mode(app: &App, frame: &mut Frame, area: Rect) {
    let content_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55),  // Protein visualization
            Constraint::Percentage(45),  // Analysis panels
        ])
        .split(area);

    // Left: protein structure
    render_protein_structure(app, frame, content_layout[0]);

    // Right: analysis panels
    let analysis_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),  // Pocket analysis
            Constraint::Length(8),   // GNN attention
            Constraint::Length(6),   // Pharmacophore
            Constraint::Min(8),      // Dialogue
        ])
        .split(content_layout[1]);

    render_pocket_analysis(app, frame, analysis_layout[0]);
    render_gnn_attention(app, frame, analysis_layout[1]);
    render_pharmacophore(app, frame, analysis_layout[2]);
    render_dialogue_history(app, frame, analysis_layout[3]);
}

/// Render welcome screen
fn render_welcome(app: &App, frame: &mut Frame, area: Rect) {
    let welcome_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60),
            Constraint::Percentage(40),
        ])
        .split(area);

    // Welcome text
    let welcome_text = vec![
        Line::from(""),
        Line::from(Span::styled("  Welcome to PRISM-Fold", Theme::title())),
        Line::from(""),
        Line::from(Span::styled("  Phase Resonance Integrated Solver Machine", Theme::dim())),
        Line::from(Span::styled("  for Molecular Folding", Theme::dim())),
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled("  Get Started:", Theme::header())),
        Line::from(""),
        Line::from("  • load <file.col>    Load a DIMACS graph"),
        Line::from("  • load <file.pdb>    Load a protein structure"),
        Line::from("  • run                Start optimization"),
        Line::from("  • help               Show all commands"),
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled("  Features:", Theme::header())),
        Line::from(""),
        Line::from("  • GPU-accelerated 7-phase optimization"),
        Line::from("  • Quantum-classical hybrid algorithms"),
        Line::from("  • GNN-based binding site prediction"),
        Line::from("  • Real-time visualization"),
        Line::from("  • AI research assistant"),
    ];

    let welcome = Paragraph::new(welcome_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::panel_border())
            .title(Span::styled(" ◆ PRISM-Fold ", Theme::panel_title())));
    frame.render_widget(welcome, welcome_layout[0]);

    // Dialogue on right
    render_dialogue_history(app, frame, welcome_layout[1]);
}

/// Render graph visualization
fn render_graph_visualization(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Live Graph Coloring ", Theme::panel_title()));

    // Use real data for graph visualization
    let current_colors = if app.optimization.colors > 0 {
        app.optimization.colors
    } else {
        app.optimization.best_colors
    };
    let current_conflicts = if app.optimization.iteration > 0 {
        app.optimization.conflicts
    } else {
        0
    };

    // Dynamic graph visualization - use actual color count to determine colors shown
    let color_indices = (0..current_colors.min(5)).collect::<Vec<_>>();
    let graph_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("        "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(0).copied().unwrap_or(0) % Theme::GRAPH_COLORS.len()])),
            Span::raw("───"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(1).copied().unwrap_or(1) % Theme::GRAPH_COLORS.len()])),
            Span::raw("───"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(2).copied().unwrap_or(2) % Theme::GRAPH_COLORS.len()])),
        ]),
        Line::from(vec![
            Span::raw("       ╱│╲   │   ╱│╲"),
        ]),
        Line::from(vec![
            Span::raw("     "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(3).copied().unwrap_or(3) % Theme::GRAPH_COLORS.len()])),
            Span::raw(" │ "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(4).copied().unwrap_or(4) % Theme::GRAPH_COLORS.len()])),
            Span::raw("─┼─"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(3).copied().unwrap_or(3) % Theme::GRAPH_COLORS.len()])),
            Span::raw(" │ "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(1).copied().unwrap_or(1) % Theme::GRAPH_COLORS.len()])),
        ]),
        Line::from(vec![
            Span::raw("      ╲│╱   │   ╲│╱"),
        ]),
        Line::from(vec![
            Span::raw("       "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(2).copied().unwrap_or(2) % Theme::GRAPH_COLORS.len()])),
            Span::raw("───"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(0).copied().unwrap_or(0) % Theme::GRAPH_COLORS.len()])),
            Span::raw("───"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[color_indices.get(4).copied().unwrap_or(4) % Theme::GRAPH_COLORS.len()])),
        ]),
        Line::from(""),
        Line::from(format!(
            "  Colors: {} │ Conflicts: {} │ Best: {}",
            current_colors,
            current_conflicts,
            if app.optimization.best_colors > 0 {
                format!("{}", app.optimization.best_colors)
            } else {
                "N/A".to_string()
            }
        )),
    ];

    let graph = Paragraph::new(graph_text).block(block);
    frame.render_widget(graph, area);
}

/// Render energy landscape
fn render_energy_landscape(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Energy Landscape ", Theme::panel_title()));

    let landscape = vec![
        Line::from(""),
        Line::from("       ╭───╮"),
        Line::from("      ╱     ╲    ╭─╮"),
        Line::from("     ╱   ◆   ╲  ╱   ╲"),
        Line::from("    ╱  here   ╲╱     ╲  ← target"),
        Line::from("   ╱           ╲      ╲"),
        Line::from("  ╱             ╲      ╲"),
        Line::from("  ─────────────────────────"),
        Line::from("    Iterations ────────→"),
    ];

    let widget = Paragraph::new(landscape).block(block);
    frame.render_widget(widget, area);
}

/// Render replica swarm
fn render_replica_swarm(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Parallel Tempering Replicas ", Theme::panel_title()));

    let mut lines = vec![Line::from("")];

    // Use real replica data if available, otherwise show placeholder
    if app.optimization.replicas.is_empty() {
        lines.push(Line::from(Span::styled("  No replicas active", Theme::dim())));
    } else {
        // Find min/max colors for normalization
        let min_colors = app.optimization.replicas.iter().map(|r| r.colors).min().unwrap_or(0);
        let max_colors = app.optimization.replicas.iter().map(|r| r.colors).max().unwrap_or(1);
        let range = (max_colors - min_colors).max(1) as f64;

        for (i, replica) in app.optimization.replicas.iter().enumerate() {
            // Normalize bar length based on color count (fewer colors = longer bar)
            let normalized = if range > 0.0 {
                1.0 - ((replica.colors - min_colors) as f64 / range)
            } else {
                0.5
            };
            let bar_len = (40.0 * normalized).max(1.0) as usize;
            let bar = "━".repeat(bar_len.min(40));

            // Color based on temperature
            let temp_color = Theme::temperature_color(i as f64 / app.optimization.replicas.len().max(1) as f64);

            let suffix = if replica.is_best { "◆ BEST" } else { "" };

            lines.push(Line::from(vec![
                Span::styled(format!("  T={:.2} ", replica.temperature), Style::default().fg(temp_color)),
                Span::styled(bar, Style::default().fg(temp_color)),
                Span::styled(format!(" {}c {}", replica.colors, suffix), Theme::normal()),
            ]));
        }
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render quantum state
fn render_quantum_state(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Quantum State ", Theme::panel_title()));

    let mut lines = vec![
        Line::from(Span::styled("  |ψ⟩ superposition:", Theme::dim())),
        Line::from(""),
    ];

    // Use real quantum amplitude data if available
    if app.optimization.quantum_amplitudes.is_empty() {
        lines.push(Line::from(Span::styled("  No quantum state", Theme::dim())));
    } else {
        // Take top 3 amplitudes
        for (color, amp) in app.optimization.quantum_amplitudes.iter().take(3) {
            let bar_len = (amp * 15.0).max(0.0).min(15.0) as usize;
            let bar = "█".repeat(bar_len) + &"░".repeat(15 - bar_len);
            lines.push(Line::from(format!("  |{}⟩ {} {:.2}", color, bar, amp)));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(format!("  coherence: {:.2}", app.optimization.quantum_coherence)));

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render dendritic activity
fn render_dendritic_activity(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Dendritic Activity ", Theme::panel_title()));

    // Use firing rate to determine how many lightning bolts to show
    let firing_intensity = (app.optimization.dendritic_firing_rate * 6.0) as usize;
    let show_bolt = |i: usize| -> Span {
        if i < firing_intensity {
            Span::styled("⚡", Style::default().fg(Theme::ACCENT))
        } else {
            Span::styled("·", Style::default().fg(Theme::TEXT_DIM))
        }
    };

    let activity = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("   "),
            show_bolt(0),
            Span::raw("    "),
            show_bolt(1),
        ]),
        Line::from("  ╱ ╲  ╱ ╲  ╭─"),
        Line::from(vec![
            Span::raw(" "),
            show_bolt(2),
            Span::raw("   "),
            show_bolt(3),
            Span::raw("    "),
            show_bolt(4),
            Span::raw("   ╲"),
        ]),
        Line::from("  ╲ ╱  ╲ ╱  ╲   "),
        Line::from(vec![
            Span::raw("   "),
            show_bolt(5),
            Span::raw("    "),
            show_bolt(0),
            Span::raw("────"),
        ]),
        Line::from(""),
        Line::from(format!(
            "  firing: {}/{} ({:.0}%)",
            app.optimization.dendritic_active_neurons,
            if app.optimization.dendritic_total_neurons > 0 {
                app.optimization.dendritic_total_neurons
            } else {
                2048
            },
            app.optimization.dendritic_firing_rate * 100.0
        )),
    ];

    let widget = Paragraph::new(activity).block(block);
    frame.render_widget(widget, area);
}

/// Render GPU kernels
fn render_gpu_kernels(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" GPU Kernels ", Theme::panel_title()));

    let mut lines = vec![Line::from("")];

    // Use real GPU kernel data if available
    if app.gpu.active_kernels.is_empty() {
        lines.push(Line::from(Span::styled("  No kernels active", Theme::dim())));
        lines.push(Line::from(""));
        lines.push(Line::from(format!("  GPU util: {:.0}%", app.gpu.utilization)));
    } else {
        // Show active kernels with utilization bars
        for (i, kernel) in app.gpu.active_kernels.iter().enumerate().take(3) {
            // Use GPU utilization for active kernels, distribute evenly
            let kernel_util = if app.gpu.active_kernels.len() > 0 {
                app.gpu.utilization / app.gpu.active_kernels.len() as f64
            } else {
                0.0
            };

            let bar_len = ((kernel_util / 10.0).min(10.0).max(0.0)) as usize;
            let bar = "█".repeat(bar_len);
            let bar_empty = "░".repeat(10 - bar_len);

            let util_color = if kernel_util > 70.0 {
                Theme::SUCCESS
            } else if kernel_util > 30.0 {
                Theme::WARNING
            } else {
                Theme::TEXT_DIM
            };

            lines.push(Line::from(format!(" {}", kernel)));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(bar, Style::default().fg(util_color)),
                Span::styled(bar_empty, Style::default().fg(Theme::TEXT_DIM)),
                Span::raw(format!(" {:.0}%", kernel_util)),
            ]));

            if i < app.gpu.active_kernels.len() - 1 && i < 2 {
                lines.push(Line::from(""));
            }
        }
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render dialogue history
fn render_dialogue_history(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Dialogue {
            Style::default().fg(Theme::ACCENT)
        } else {
            Theme::panel_border()
        })
        .title(Span::styled(" AI Assistant ", Theme::panel_title()));

    let messages: Vec<Line> = app.dialogue.messages.iter()
        .flat_map(|msg| {
            let style = if msg.is_user {
                Style::default().fg(Theme::ACCENT_SECONDARY)
            } else {
                Theme::normal()
            };
            let prefix = if msg.is_user { "> " } else { "  " };

            msg.content.lines()
                .map(|line| Line::from(Span::styled(format!("{}{}", prefix, line), style)))
                .collect::<Vec<_>>()
        })
        .collect();

    let widget = Paragraph::new(messages)
        .block(block)
        .wrap(Wrap { trim: true })
        .scroll((app.dialogue.scroll_offset as u16, 0));
    frame.render_widget(widget, area);
}

/// Render pipeline flow
fn render_pipeline_flow(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Phase Pipeline ", Theme::panel_title()));

    let mut lines = vec![Line::from("")];

    // Phase flow line
    let mut phase_spans = vec![Span::raw("  ")];
    for (i, phase) in app.phases.iter().enumerate() {
        let (symbol, style) = match phase.status {
            PhaseState::Completed => ("✓", Theme::success()),
            PhaseState::Running => ("▶", Style::default().fg(Theme::ACCENT)),
            PhaseState::Failed => ("✗", Theme::error()),
            PhaseState::Pending => ("○", Theme::dim()),
        };

        phase_spans.push(Span::styled(symbol, style));
        if i < app.phases.len() - 1 {
            phase_spans.push(Span::styled("─", Theme::dim()));
        }
    }
    lines.push(Line::from(phase_spans));

    // Phase names
    let names: String = app.phases.iter()
        .map(|p| format!("{:^3}", &p.name[..2]))
        .collect::<Vec<_>>()
        .join("");
    lines.push(Line::from(Span::styled(format!("  {}", names), Theme::dim())));

    // Progress bar for running phase
    if let Some(phase) = app.phases.iter().find(|p| p.status == PhaseState::Running) {
        let bar_width = 30;
        let filled = ((phase.progress / 100.0) * bar_width as f64) as usize;
        let bar = format!(
            "  [{}{}] {:.0}%",
            "█".repeat(filled),
            "░".repeat(bar_width - filled),
            phase.progress
        );
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(bar, Style::default().fg(Theme::progress_color(phase.progress)))));
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render convergence chart
fn render_convergence_chart(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Convergence ", Theme::panel_title()));

    // Use real convergence history data
    let mut chart = Vec::new();

    if app.optimization.convergence_history.is_empty() {
        chart.push(Line::from(""));
        chart.push(Line::from(Span::styled("  No data yet", Theme::dim())));
        chart.push(Line::from(""));
    } else {
        // Find min/max colors for scaling
        let colors: Vec<usize> = app.optimization.convergence_history.iter().map(|(_, c)| *c).collect();
        let min_colors = *colors.iter().min().unwrap_or(&0);
        let max_colors = *colors.iter().max().unwrap_or(&100);
        let range = (max_colors - min_colors).max(1);

        // Create 5 rows for the chart (from max to min)
        let chart_height = 4;
        let chart_width = 20;

        for row in 0..=chart_height {
            let threshold = max_colors - (row * range / chart_height);
            let mut line_spans = vec![
                Span::raw(format!(" {:3} ", threshold)),
                if row == chart_height { Span::raw("└") } else { Span::raw("┤") },
            ];

            // Plot points
            let sample_step = (app.optimization.convergence_history.len()).max(1) / chart_width.min(app.optimization.convergence_history.len());
            let sample_step = sample_step.max(1);

            for i in (0..app.optimization.convergence_history.len()).step_by(sample_step).take(chart_width) {
                let (_, color_count) = app.optimization.convergence_history[i];
                let y_pos = ((max_colors - color_count) * chart_height) / range.max(1);

                if y_pos == row {
                    // Is this the best/latest point?
                    let is_best = i == app.optimization.convergence_history.len() - 1
                                   && color_count == min_colors;
                    if is_best {
                        line_spans.push(Span::styled("◆", Style::default().fg(Theme::SUCCESS)));
                    } else {
                        line_spans.push(Span::raw("•"));
                    }
                } else {
                    line_spans.push(Span::raw(" "));
                }
            }

            chart.push(Line::from(line_spans));
        }

        // Add x-axis markers
        chart.push(Line::from("      iterations ───→"));
    }

    let widget = Paragraph::new(chart).block(block);
    frame.render_widget(widget, area);
}

/// Render protein structure (biomolecular mode)
fn render_protein_structure(app: &App, frame: &mut Frame, area: Rect) {
    let title = if app.protein.name.is_empty() {
        " Protein Structure ".to_string()
    } else {
        format!(" {} ", app.protein.name)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(title, Theme::panel_title()));

    let structure = vec![
        Line::from(""),
        Line::from("                    ┌───── Active Site ─────┐"),
        Line::from(vec![
            Span::raw("                   ╱    "),
            Span::styled("● High druggability", Style::default().fg(Theme::ERROR)),
            Span::raw(" ╲"),
        ]),
        Line::from(vec![
            Span::raw("       "),
            Span::styled("╭━━━━━━━╮", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw("  ╱                          ╲  "),
            Span::styled("╭━━━━━━━╮", Style::default().fg(Theme::PROTEIN_HELIX)),
        ]),
        Line::from(vec![
            Span::styled(" ░░░░░░┃██████┃", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw("━━        ╭─── Ligand ───╮      "),
            Span::styled("┃██████┃░░░░░░", Style::default().fg(Theme::PROTEIN_HELIX)),
        ]),
        Line::from(vec![
            Span::raw("       "),
            Span::styled("┃██████┃", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw("    ░░░░░░│   "),
            Span::styled("⬡     ⬡", Style::default().fg(Theme::PROTEIN_LIGAND)),
            Span::raw("    │░░░░░░"),
            Span::styled("┃██████┃", Style::default().fg(Theme::PROTEIN_HELIX)),
        ]),
        Line::from(vec![
            Span::raw("       "),
            Span::styled("╰━━━━━━━╯", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw("  ╲░░░░░░│  "),
            Span::styled("⬡─⬡───⬡─⬡", Style::default().fg(Theme::PROTEIN_LIGAND)),
            Span::raw("   │░░░░░╱"),
            Span::styled("╰━━━━━━━╯", Style::default().fg(Theme::PROTEIN_HELIX)),
        ]),
        Line::from("          │        ╲░░░░░╰──────────────╯░░░╱        │"),
        Line::from("     "),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("┌─────────┐", Style::default().fg(Theme::PROTEIN_SHEET)),
            Span::raw("     ╲░░░░░ Pocket ░░░░░╱     "),
            Span::styled("┌─────────┐", Style::default().fg(Theme::PROTEIN_SHEET)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("│▓▓▓▓▓▓▓▓▓│", Style::default().fg(Theme::PROTEIN_SHEET)),
            Span::raw("         β-sheet         "),
            Span::styled("│▓▓▓▓▓▓▓▓▓│", Style::default().fg(Theme::PROTEIN_SHEET)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("└─────────┘", Style::default().fg(Theme::PROTEIN_SHEET)),
            Span::raw("                          "),
            Span::styled("└─────────┘", Style::default().fg(Theme::PROTEIN_SHEET)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("━━━", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw(" α-helix  "),
            Span::styled("▓▓▓", Style::default().fg(Theme::PROTEIN_SHEET)),
            Span::raw(" β-sheet  "),
            Span::styled("░░░", Style::default().fg(Theme::PROTEIN_POCKET)),
            Span::raw(" pocket  "),
            Span::styled("⬡", Style::default().fg(Theme::PROTEIN_LIGAND)),
            Span::raw(" ligand"),
        ]),
    ];

    // Add real protein statistics at bottom if available
    let mut final_structure = structure;
    if app.protein.residue_count > 0 {
        final_structure.push(Line::from(""));
        final_structure.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!(
                    "{} residues │ {} atoms │ {} chains",
                    app.protein.residue_count,
                    app.protein.atom_count,
                    app.protein.chain_count
                ),
                Style::default().fg(Theme::TEXT_DIM)
            ),
        ]));
    }

    let widget = Paragraph::new(final_structure).block(block);
    frame.render_widget(widget, area);
}

/// Render pocket analysis
fn render_pocket_analysis(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(
            format!(" Detected Pockets ({}) ", app.protein.pockets.len()),
            Theme::panel_title()
        ));

    let mut lines = vec![];

    if app.protein.pockets.is_empty() {
        // Show LBS progress if detection is ongoing
        if let Some(ref phase) = app.protein.lbs_progress.current_phase {
            lines.push(Line::from(""));
            lines.push(Line::from(format!("  Phase: {}", phase)));
            lines.push(Line::from(format!(
                "  Progress: {}/{}",
                app.protein.lbs_progress.phase_iteration,
                app.protein.lbs_progress.phase_max_iterations
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(format!(
                "  Pockets found: {}",
                app.protein.lbs_progress.pockets_detected
            )));
            if app.protein.lbs_progress.best_druggability > 0.0 {
                lines.push(Line::from(format!(
                    "  Best druggability: {:.3}",
                    app.protein.lbs_progress.best_druggability
                )));
            }
            if app.protein.lbs_progress.gpu_accelerated {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::raw("  GPU: "),
                    Span::styled("ACTIVE", Style::default().fg(Theme::SUCCESS)),
                ]));
            }
        } else {
            lines.push(Line::from(""));
            lines.push(Line::from("  No pockets detected yet"));
            lines.push(Line::from(""));
            lines.push(Line::from("  Awaiting structure load..."));
        }
    } else {
        // Show top 5 pockets with real data
        for (i, pocket) in app.protein.pockets.iter().take(5).enumerate() {
            if i > 0 {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(""));

            // Pocket header
            let drug_color = if pocket.druggability > 0.8 {
                Theme::SUCCESS
            } else if pocket.druggability > 0.5 {
                Theme::WARNING
            } else {
                Theme::TEXT_DIM
            };

            lines.push(Line::from(vec![
                Span::raw(format!("  #{} ", pocket.id)),
                Span::styled(
                    format!("Vol: {:.1}Å³  Drug: {:.2}", pocket.volume, pocket.druggability),
                    Style::default().fg(drug_color).bold()
                ),
            ]));

            // Druggability bar
            let bar_len = (pocket.druggability * 10.0) as usize;
            let bar = "█".repeat(bar_len) + &"░".repeat(10 - bar_len);
            lines.push(Line::from(vec![
                Span::raw("     "),
                Span::styled(bar, Style::default().fg(drug_color)),
            ]));

            // Residues (show first 5)
            let residue_list: String = pocket.residues.iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            let residue_display = if pocket.residues.len() > 5 {
                format!("{}...", residue_list)
            } else {
                residue_list
            };
            lines.push(Line::from(vec![
                Span::raw("     "),
                Span::styled(residue_display, Style::default().fg(Theme::TEXT_DIM)),
            ]));
        }
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render GNN attention
fn render_gnn_attention(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" GNN Attention Map ", Theme::panel_title()));

    let mut lines = vec![Line::from("")];

    if app.protein.gnn_attention.is_empty() {
        lines.push(Line::from("  No GNN data available"));
        lines.push(Line::from(""));
        lines.push(Line::from("  Run prediction to see"));
        lines.push(Line::from("  residue importance"));
    } else {
        // Show top attention scores from real data
        for (res, attn) in app.protein.gnn_attention.iter().take(4) {
            let bar_len = (attn * 20.0) as usize;
            let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
            let color = if *attn > 0.8 {
                Theme::SUCCESS
            } else if *attn > 0.5 {
                Theme::WARNING
            } else {
                Theme::TEXT_DIM
            };
            lines.push(Line::from(vec![
                Span::raw(format!("  {:8} ", res)),
                Span::styled(bar, Style::default().fg(color)),
                Span::raw(format!(" {:.2}", attn)),
            ]));
        }
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render pharmacophore features
fn render_pharmacophore(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Pharmacophore Features ", Theme::panel_title()));

    let features = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", Style::default().fg(Theme::PHARM_DONOR)),
            Span::raw(" H-bond Donor    ×4  "),
            Span::styled("●", Style::default().fg(Theme::PHARM_ACCEPTOR)),
            Span::raw(" H-bond Acceptor ×6"),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", Style::default().fg(Theme::PHARM_HYDROPHOBIC)),
            Span::raw(" Hydrophobic     ×3  "),
            Span::styled("●", Style::default().fg(Theme::PHARM_AROMATIC)),
            Span::raw(" Aromatic        ×2"),
        ]),
    ];

    let widget = Paragraph::new(features).block(block);
    frame.render_widget(widget, area);
}

/// Render dialogue input bar
fn render_dialogue_input(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Dialogue {
            Style::default().fg(Theme::ACCENT)
        } else {
            Theme::panel_border()
        });

    let input_text = format!(" > {}_", app.input_buffer);
    let input = Paragraph::new(input_text)
        .style(Theme::normal())
        .block(block);
    frame.render_widget(input, area);
}

/// Render help overlay
fn render_help_overlay(frame: &mut Frame, area: Rect) {
    let help_area = centered_rect(60, 70, area);

    // Clear background
    frame.render_widget(Clear, help_area);

    let help_text = vec![
        Line::from(Span::styled(" Keyboard Shortcuts ", Theme::title())),
        Line::from(""),
        Line::from(" Navigation:"),
        Line::from("   Tab        Cycle focus between panels"),
        Line::from("   Esc        Close overlays / cancel"),
        Line::from("   F1         Toggle this help"),
        Line::from(""),
        Line::from(" Commands (type in dialogue):"),
        Line::from("   load <file>    Load graph or protein"),
        Line::from("   run / go       Start optimization"),
        Line::from("   stop / pause   Pause optimization"),
        Line::from("   status         Show current status"),
        Line::from("   set <p> <v>    Set parameter"),
        Line::from("   help           Show command help"),
        Line::from(""),
        Line::from(" Quick Actions:"),
        Line::from("   Ctrl+C/Q   Quit"),
        Line::from("   g          Focus graph view"),
        Line::from("   p          Focus protein view"),
        Line::from("   m          Focus metrics"),
    ];

    let help = Paragraph::new(help_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::ACCENT))
            .title(Span::styled(" Help ", Theme::panel_title()))
            .style(Style::default().bg(Theme::BG_PANEL)));

    frame.render_widget(help, help_area);
}

/// Helper to create a centered rect
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
