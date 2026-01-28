//! PRISM Color Theme
//!
//! World-class color palette for the TUI.

use ratatui::style::{Color, Modifier, Style};

/// PRISM color theme - bleeding edge aesthetics
pub struct Theme;

impl Theme {
    // ══════════════════════════════════════════════════════════════════════
    // Primary Colors
    // ══════════════════════════════════════════════════════════════════════

    /// Primary accent color - electric cyan
    pub const ACCENT: Color = Color::Rgb(0, 255, 255);

    /// Secondary accent - warm gold
    pub const ACCENT_SECONDARY: Color = Color::Rgb(255, 215, 0);

    /// Success/valid - emerald green
    pub const SUCCESS: Color = Color::Rgb(0, 255, 127);

    /// Warning - amber
    pub const WARNING: Color = Color::Rgb(255, 191, 0);

    /// Error/conflict - coral red
    pub const ERROR: Color = Color::Rgb(255, 99, 71);

    /// Info - sky blue
    pub const INFO: Color = Color::Rgb(135, 206, 250);

    // ══════════════════════════════════════════════════════════════════════
    // Background Colors
    // ══════════════════════════════════════════════════════════════════════

    /// Main background - deep space black
    pub const BG_PRIMARY: Color = Color::Rgb(13, 17, 23);

    /// Panel background - slightly lighter
    pub const BG_PANEL: Color = Color::Rgb(22, 27, 34);

    /// Highlighted background
    pub const BG_HIGHLIGHT: Color = Color::Rgb(33, 38, 45);

    /// Selection background
    pub const BG_SELECTION: Color = Color::Rgb(48, 54, 61);

    // ══════════════════════════════════════════════════════════════════════
    // Text Colors
    // ══════════════════════════════════════════════════════════════════════

    /// Primary text - bright white
    pub const TEXT_PRIMARY: Color = Color::Rgb(240, 246, 252);

    /// Secondary text - muted gray
    pub const TEXT_SECONDARY: Color = Color::Rgb(139, 148, 158);

    /// Dimmed text
    pub const TEXT_DIM: Color = Color::Rgb(110, 118, 129);

    // ══════════════════════════════════════════════════════════════════════
    // Graph Coloring Palette (22 distinct colors)
    // ══════════════════════════════════════════════════════════════════════

    pub const GRAPH_COLORS: [Color; 22] = [
        Color::Rgb(255, 99, 71),   // Tomato
        Color::Rgb(255, 215, 0),   // Gold
        Color::Rgb(50, 205, 50),   // Lime Green
        Color::Rgb(30, 144, 255),  // Dodger Blue
        Color::Rgb(186, 85, 211),  // Medium Orchid
        Color::Rgb(255, 127, 80),  // Coral
        Color::Rgb(0, 255, 255),   // Cyan
        Color::Rgb(255, 20, 147),  // Deep Pink
        Color::Rgb(127, 255, 0),   // Chartreuse
        Color::Rgb(255, 165, 0),   // Orange
        Color::Rgb(138, 43, 226),  // Blue Violet
        Color::Rgb(0, 250, 154),   // Medium Spring Green
        Color::Rgb(255, 69, 0),    // Red Orange
        Color::Rgb(70, 130, 180),  // Steel Blue
        Color::Rgb(255, 182, 193), // Light Pink
        Color::Rgb(144, 238, 144), // Light Green
        Color::Rgb(255, 228, 181), // Moccasin
        Color::Rgb(176, 196, 222), // Light Steel Blue
        Color::Rgb(221, 160, 221), // Plum
        Color::Rgb(152, 251, 152), // Pale Green
        Color::Rgb(240, 128, 128), // Light Coral
        Color::Rgb(175, 238, 238), // Pale Turquoise
    ];

    // ══════════════════════════════════════════════════════════════════════
    // Temperature Gradient (for parallel tempering)
    // ══════════════════════════════════════════════════════════════════════

    pub const TEMP_COLD: Color = Color::Rgb(59, 130, 246);   // Blue
    pub const TEMP_COOL: Color = Color::Rgb(34, 197, 94);    // Green
    pub const TEMP_WARM: Color = Color::Rgb(250, 204, 21);   // Yellow
    pub const TEMP_HOT: Color = Color::Rgb(249, 115, 22);    // Orange
    pub const TEMP_FIRE: Color = Color::Rgb(239, 68, 68);    // Red
    pub const TEMP_PLASMA: Color = Color::Rgb(168, 85, 247); // Purple

    // ══════════════════════════════════════════════════════════════════════
    // Protein Visualization
    // ══════════════════════════════════════════════════════════════════════

    /// Alpha helix - red/magenta
    pub const PROTEIN_HELIX: Color = Color::Rgb(255, 0, 128);

    /// Beta sheet - yellow/gold
    pub const PROTEIN_SHEET: Color = Color::Rgb(255, 215, 0);

    /// Loop/coil - gray
    pub const PROTEIN_LOOP: Color = Color::Rgb(169, 169, 169);

    /// Binding pocket - green glow
    pub const PROTEIN_POCKET: Color = Color::Rgb(0, 255, 127);

    /// Ligand - cyan
    pub const PROTEIN_LIGAND: Color = Color::Rgb(0, 255, 255);

    // ══════════════════════════════════════════════════════════════════════
    // Pharmacophore Features
    // ══════════════════════════════════════════════════════════════════════

    pub const PHARM_DONOR: Color = Color::Rgb(255, 99, 71);    // Red - H-bond donor
    pub const PHARM_ACCEPTOR: Color = Color::Rgb(30, 144, 255); // Blue - H-bond acceptor
    pub const PHARM_HYDROPHOBIC: Color = Color::Rgb(255, 215, 0); // Yellow
    pub const PHARM_AROMATIC: Color = Color::Rgb(186, 85, 211);  // Purple
    pub const PHARM_METAL: Color = Color::Rgb(192, 192, 192);    // Silver

    // ══════════════════════════════════════════════════════════════════════
    // GPU Status
    // ══════════════════════════════════════════════════════════════════════

    pub const GPU_UTILIZATION_LOW: Color = Color::Rgb(34, 197, 94);
    pub const GPU_UTILIZATION_MED: Color = Color::Rgb(250, 204, 21);
    pub const GPU_UTILIZATION_HIGH: Color = Color::Rgb(249, 115, 22);
    pub const GPU_UTILIZATION_MAX: Color = Color::Rgb(239, 68, 68);

    // ══════════════════════════════════════════════════════════════════════
    // Style Helpers
    // ══════════════════════════════════════════════════════════════════════

    pub fn title() -> Style {
        Style::default()
            .fg(Self::ACCENT)
            .add_modifier(Modifier::BOLD)
    }

    pub fn header() -> Style {
        Style::default()
            .fg(Self::TEXT_PRIMARY)
            .add_modifier(Modifier::BOLD)
    }

    pub fn normal() -> Style {
        Style::default().fg(Self::TEXT_PRIMARY)
    }

    pub fn dim() -> Style {
        Style::default().fg(Self::TEXT_DIM)
    }

    pub fn success() -> Style {
        Style::default().fg(Self::SUCCESS)
    }

    pub fn warning() -> Style {
        Style::default().fg(Self::WARNING)
    }

    pub fn error() -> Style {
        Style::default().fg(Self::ERROR)
    }

    pub fn highlight() -> Style {
        Style::default()
            .fg(Self::ACCENT)
            .add_modifier(Modifier::BOLD)
    }

    pub fn panel_border() -> Style {
        Style::default().fg(Self::TEXT_DIM)
    }

    pub fn panel_title() -> Style {
        Style::default()
            .fg(Self::ACCENT_SECONDARY)
            .add_modifier(Modifier::BOLD)
    }

    /// Get color for graph vertex by color index
    pub fn graph_color(index: usize) -> Color {
        Self::GRAPH_COLORS[index % Self::GRAPH_COLORS.len()]
    }

    /// Get color for temperature value (0.0 = cold, 1.0 = hot)
    pub fn temperature_color(normalized: f64) -> Color {
        if normalized < 0.2 {
            Self::TEMP_COLD
        } else if normalized < 0.4 {
            Self::TEMP_COOL
        } else if normalized < 0.6 {
            Self::TEMP_WARM
        } else if normalized < 0.8 {
            Self::TEMP_HOT
        } else {
            Self::TEMP_FIRE
        }
    }

    /// Get color for GPU utilization (0-100)
    pub fn gpu_util_color(percent: f64) -> Color {
        if percent < 50.0 {
            Self::GPU_UTILIZATION_LOW
        } else if percent < 75.0 {
            Self::GPU_UTILIZATION_MED
        } else if percent < 90.0 {
            Self::GPU_UTILIZATION_HIGH
        } else {
            Self::GPU_UTILIZATION_MAX
        }
    }

    /// Get color for progress bar based on completion
    pub fn progress_color(percent: f64) -> Color {
        if percent < 33.0 {
            Self::INFO
        } else if percent < 66.0 {
            Self::WARNING
        } else {
            Self::SUCCESS
        }
    }
}
