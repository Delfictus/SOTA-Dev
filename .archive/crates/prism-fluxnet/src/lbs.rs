//! LBS (Load-Balanced Scheduling) Extensions for FluxNet
//!
//! This module provides LBS-specific state extensions and reward shaping
//! for adaptive Phase 2 thermodynamic equilibration.
//!
//! # LBS Integration
//!
//! The Load-Balanced Scheduling (LBS) system distributes Phase 2 workload
//! across multiple GPU devices or CPU threads. FluxNet RL learns to:
//! - Balance workload distribution
//! - Optimize batch sizes per device
//! - Adapt scheduling strategies based on conflict patterns
//!
//! # State Extensions
//!
//! LBS extends UniversalRLState with:
//! - `lbs_load_imbalance`: Variance in workload distribution (0.0 = perfect, 1.0 = severe)
//! - `lbs_throughput`: Operations per second (normalized)
//! - `lbs_active_workers`: Number of active GPU/CPU workers
//! - `lbs_queue_depth`: Average task queue depth

use serde::{Deserialize, Serialize};

/// LBS-specific state extensions for FluxNet RL
///
/// These metrics capture load-balancing performance and inform
/// adaptive scheduling decisions during Phase 2 thermodynamic equilibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LbsState {
    /// Load imbalance metric: stddev / mean of worker utilization
    /// Range: [0.0, 1.0] where 0.0 = perfectly balanced, 1.0 = severely imbalanced
    pub load_imbalance: f32,

    /// Throughput: operations per second (normalized by max capacity)
    /// Range: [0.0, 1.0] where 1.0 = maximum throughput
    pub throughput: f32,

    /// Number of active workers (GPUs or CPU threads)
    pub active_workers: usize,

    /// Average queue depth across all workers
    /// Lower is better (less queueing = better load distribution)
    pub avg_queue_depth: f32,

    /// Peak worker utilization (0.0 = idle, 1.0 = saturated)
    pub peak_utilization: f32,

    /// Minimum worker utilization
    pub min_utilization: f32,

    /// Number of task migrations (rebalancing events)
    pub migration_count: usize,
}

impl Default for LbsState {
    fn default() -> Self {
        Self {
            load_imbalance: 0.0,
            throughput: 0.0,
            active_workers: 1,
            avg_queue_depth: 0.0,
            peak_utilization: 0.0,
            min_utilization: 0.0,
            migration_count: 0,
        }
    }
}

impl LbsState {
    /// Create a new LBS state from worker metrics
    ///
    /// # Arguments
    /// - `worker_utilizations`: Per-worker utilization ratios [0.0, 1.0]
    /// - `throughput`: Current operations per second (normalized)
    /// - `queue_depths`: Per-worker queue depths
    pub fn new(
        worker_utilizations: &[f32],
        throughput: f32,
        queue_depths: &[f32],
        migration_count: usize,
    ) -> Self {
        if worker_utilizations.is_empty() {
            return Self::default();
        }

        // Compute load imbalance (coefficient of variation)
        let mean = worker_utilizations.iter().sum::<f32>() / worker_utilizations.len() as f32;
        let variance = worker_utilizations
            .iter()
            .map(|&u| (u - mean).powi(2))
            .sum::<f32>()
            / worker_utilizations.len() as f32;
        let stddev = variance.sqrt();
        let load_imbalance = if mean > 0.0 { (stddev / mean).min(1.0) } else { 0.0 };

        // Peak and min utilization
        let peak_utilization = worker_utilizations.iter().cloned().fold(0.0f32, f32::max);
        let min_utilization = worker_utilizations.iter().cloned().fold(1.0f32, f32::min);

        // Average queue depth
        let avg_queue_depth = if queue_depths.is_empty() {
            0.0
        } else {
            queue_depths.iter().sum::<f32>() / queue_depths.len() as f32
        };

        Self {
            load_imbalance,
            throughput: throughput.clamp(0.0, 1.0),
            active_workers: worker_utilizations.len(),
            avg_queue_depth,
            peak_utilization,
            min_utilization,
            migration_count,
        }
    }

    /// Compute LBS reward bonus for FluxNet RL
    ///
    /// # Objectives
    /// - Minimize load imbalance (reward balanced distribution)
    /// - Maximize throughput (reward efficient scheduling)
    /// - Minimize queue depth (reward low latency)
    /// - Minimize migrations (reward stable assignments)
    ///
    /// # Returns
    /// Reward bonus in range [-0.5, +0.5]
    pub fn compute_reward_bonus(&self, previous: &LbsState) -> f32 {
        // Load imbalance reduction (lower is better)
        let imbalance_delta = previous.load_imbalance - self.load_imbalance;
        let imbalance_reward = (imbalance_delta * 0.5).clamp(-0.2, 0.2);

        // Throughput improvement (higher is better)
        let throughput_delta = self.throughput - previous.throughput;
        let throughput_reward = (throughput_delta * 0.3).clamp(-0.2, 0.2);

        // Queue depth reduction (lower is better)
        let queue_delta = previous.avg_queue_depth - self.avg_queue_depth;
        let queue_reward = (queue_delta * 0.1).clamp(-0.1, 0.1);

        // Migration penalty (fewer migrations = more stable)
        let migration_delta = (self.migration_count as f32 - previous.migration_count as f32) / 10.0;
        let migration_penalty = (-migration_delta).clamp(-0.1, 0.0);

        // Combined reward
        imbalance_reward + throughput_reward + queue_reward + migration_penalty
    }

    /// Check if load is severely imbalanced (triggers rebalancing)
    pub fn is_severely_imbalanced(&self) -> bool {
        self.load_imbalance > 0.3 // >30% coefficient of variation
    }

    /// Check if throughput is degraded (below threshold)
    pub fn is_throughput_degraded(&self, threshold: f32) -> bool {
        self.throughput < threshold
    }

    /// Get utilization spread (max - min)
    pub fn utilization_spread(&self) -> f32 {
        self.peak_utilization - self.min_utilization
    }
}

/// LBS action space for FluxNet RL
///
/// These actions adjust load-balancing parameters during Phase 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LbsAction {
    /// Increase batch size per worker (more work per task)
    IncreaseBatchSize,

    /// Decrease batch size per worker (finer granularity)
    DecreaseBatchSize,

    /// Enable work stealing (allow idle workers to steal tasks)
    EnableWorkStealing,

    /// Disable work stealing (static assignment)
    DisableWorkStealing,

    /// Rebalance workload (migrate tasks from overloaded to idle workers)
    TriggerRebalance,

    /// Add a worker (if resources available)
    AddWorker,

    /// Remove a worker (if underutilized)
    RemoveWorker,

    /// No operation (maintain current configuration)
    NoOp,
}

impl LbsAction {
    /// Total number of LBS actions
    pub const ACTION_SPACE_SIZE: usize = 8;

    /// Convert action to index for Q-table lookup
    pub fn to_index(&self) -> usize {
        match self {
            LbsAction::IncreaseBatchSize => 0,
            LbsAction::DecreaseBatchSize => 1,
            LbsAction::EnableWorkStealing => 2,
            LbsAction::DisableWorkStealing => 3,
            LbsAction::TriggerRebalance => 4,
            LbsAction::AddWorker => 5,
            LbsAction::RemoveWorker => 6,
            LbsAction::NoOp => 7,
        }
    }

    /// Convert index to action
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => LbsAction::IncreaseBatchSize,
            1 => LbsAction::DecreaseBatchSize,
            2 => LbsAction::EnableWorkStealing,
            3 => LbsAction::DisableWorkStealing,
            4 => LbsAction::TriggerRebalance,
            5 => LbsAction::AddWorker,
            6 => LbsAction::RemoveWorker,
            _ => LbsAction::NoOp,
        }
    }

    /// Get all LBS actions
    pub fn all_actions() -> Vec<Self> {
        vec![
            LbsAction::IncreaseBatchSize,
            LbsAction::DecreaseBatchSize,
            LbsAction::EnableWorkStealing,
            LbsAction::DisableWorkStealing,
            LbsAction::TriggerRebalance,
            LbsAction::AddWorker,
            LbsAction::RemoveWorker,
            LbsAction::NoOp,
        ]
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            LbsAction::IncreaseBatchSize => "Increase batch size per worker",
            LbsAction::DecreaseBatchSize => "Decrease batch size per worker",
            LbsAction::EnableWorkStealing => "Enable work stealing",
            LbsAction::DisableWorkStealing => "Disable work stealing",
            LbsAction::TriggerRebalance => "Trigger workload rebalancing",
            LbsAction::AddWorker => "Add worker (if available)",
            LbsAction::RemoveWorker => "Remove underutilized worker",
            LbsAction::NoOp => "No operation",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbs_state_creation() {
        let utilizations = vec![0.8, 0.9, 0.7, 0.85];
        let queue_depths = vec![2.0, 3.0, 1.5, 2.5];
        let state = LbsState::new(&utilizations, 0.75, &queue_depths, 0);

        assert!(state.load_imbalance >= 0.0 && state.load_imbalance <= 1.0);
        assert_eq!(state.active_workers, 4);
        assert!(state.avg_queue_depth > 0.0);
    }

    #[test]
    fn test_lbs_reward_computation() {
        let prev_state = LbsState {
            load_imbalance: 0.5,
            throughput: 0.6,
            avg_queue_depth: 5.0,
            migration_count: 2,
            ..Default::default()
        };

        let curr_state = LbsState {
            load_imbalance: 0.3, // Improved
            throughput: 0.7,     // Improved
            avg_queue_depth: 3.0, // Improved
            migration_count: 2,  // Stable
            ..Default::default()
        };

        let reward = curr_state.compute_reward_bonus(&prev_state);
        assert!(reward > 0.0, "Reward should be positive for improvements");
    }

    #[test]
    fn test_lbs_action_indexing() {
        for action in LbsAction::all_actions() {
            let idx = action.to_index();
            let reconstructed = LbsAction::from_index(idx);
            assert_eq!(action, reconstructed, "Action round-trip should be consistent");
        }
    }

    #[test]
    fn test_severe_imbalance_detection() {
        let state = LbsState {
            load_imbalance: 0.4,
            ..Default::default()
        };
        assert!(state.is_severely_imbalanced());

        let balanced_state = LbsState {
            load_imbalance: 0.1,
            ..Default::default()
        };
        assert!(!balanced_state.is_severely_imbalanced());
    }
}
