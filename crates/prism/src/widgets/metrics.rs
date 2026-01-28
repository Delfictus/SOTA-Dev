//! Metrics visualization widgets
//!
//! Real-time charts, progress bars, and status displays.

/// Sparkline data for mini charts
pub struct Sparkline {
    pub data: Vec<f64>,
    pub max_points: usize,
}

impl Sparkline {
    pub fn new(max_points: usize) -> Self {
        Self {
            data: Vec::with_capacity(max_points),
            max_points,
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.data.len() >= self.max_points {
            self.data.remove(0);
        }
        self.data.push(value);
    }

    pub fn to_bars(&self, height: usize) -> Vec<char> {
        let bar_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if self.data.is_empty() {
            return vec![];
        }

        let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max - min;

        self.data.iter().map(|&v| {
            if range == 0.0 {
                bar_chars[bar_chars.len() / 2]
            } else {
                let normalized = (v - min) / range;
                let index = (normalized * (bar_chars.len() - 1) as f64) as usize;
                bar_chars[index.min(bar_chars.len() - 1)]
            }
        }).collect()
    }
}

/// Progress tracking
pub struct ProgressTracker {
    pub current: f64,
    pub total: f64,
    pub label: String,
    pub history: Sparkline,
}

impl ProgressTracker {
    pub fn new(label: &str, total: f64) -> Self {
        Self {
            current: 0.0,
            total,
            label: label.to_string(),
            history: Sparkline::new(50),
        }
    }

    pub fn update(&mut self, current: f64) {
        self.current = current;
        self.history.push(current / self.total * 100.0);
    }

    pub fn percent(&self) -> f64 {
        if self.total == 0.0 {
            0.0
        } else {
            (self.current / self.total * 100.0).min(100.0)
        }
    }
}

impl Default for Sparkline {
    fn default() -> Self {
        Self::new(50)
    }
}
