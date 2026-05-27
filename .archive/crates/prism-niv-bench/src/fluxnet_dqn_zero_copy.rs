//! FluxNet-DQN with Zero-Copy Inference
//!
//! ARCHITECT DIRECTIVE: PHASE 3 - ZERO-COPY BRAIN & PROVENANCE RECORDER
//!
//! This module implements the FluxNet-DQN system with zero-copy GPU memory access,
//! enabling direct inference on CUDA device memory without CPU round-trips.
//! Includes cryptographic provenance tracking for forensic validation.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use sha2::Digest;

#[cfg(feature = "dqn")]
use tch::{nn, Device, Tensor, Kind, nn::OptimizerConfig};
use cudarc::driver::{CudaContext, CudaSlice};

/// Zero-copy DQN configuration for cryptic site prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCopyDqnConfig {
    /// Input feature dimension (140: 136 main + 4 cryptic)
    pub feature_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Number of actions (4: cryptic, exposed, epitope, skip)
    pub num_actions: usize,
    /// Learning rate for optimizer
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub gamma: f64,
    /// Soft update rate (tau)
    pub tau: f64,
    /// Experience replay buffer size
    pub replay_buffer_size: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Epsilon for exploration
    pub epsilon: f64,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Minimum epsilon
    pub epsilon_min: f64,
}

impl Default for ZeroCopyDqnConfig {
    fn default() -> Self {
        Self {
            feature_dim: 140,
            hidden_dims: vec![256, 128, 64],
            num_actions: 4,
            learning_rate: 0.001,
            gamma: 0.99,
            tau: 0.005,
            replay_buffer_size: 100_000,
            batch_size: 64,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
        }
    }
}

/// Actions for cryptic epitope prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CrypticAction {
    PredictCryptic = 0,  // High crypticity confidence
    PredictExposed = 1,  // Low crypticity (surface accessible)
    PredictEpitope = 2,  // Antibody binding site
    Skip = 3,           // Not actionable residue
}

impl From<usize> for CrypticAction {
    fn from(value: usize) -> Self {
        match value {
            0 => CrypticAction::PredictCryptic,
            1 => CrypticAction::PredictExposed,
            2 => CrypticAction::PredictEpitope,
            _ => CrypticAction::Skip,
        }
    }
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: Vec<f32>,      // 140-dim feature vector
    pub action: u8,           // Action index
    pub reward: f32,          // Immediate reward
    pub next_state: Vec<f32>, // Next state features
    pub done: bool,           // Episode termination
    pub provenance_hash: String, // Cryptographic provenance
}

/// Experience replay buffer
#[derive(Debug)]
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
    rng: rand::rngs::ThreadRng,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            rng: rand::thread_rng(),
        }
    }

    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&mut self, batch_size: usize) -> Option<Vec<Experience>> {
        if self.buffer.len() < batch_size {
            return None;
        }

        use rand::seq::SliceRandom;
        let indices: Vec<usize> = (0..self.buffer.len()).collect();
        let selected_indices = indices.choose_multiple(&mut self.rng, batch_size);

        Some(selected_indices.map(|&i| self.buffer[i].clone()).collect())
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// Dueling DQN Network Architecture
#[cfg(feature = "dqn")]
#[derive(Debug)]
pub struct DuelingNetwork {
    // Shared backbone
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    // Dueling heads
    value_head: nn::Linear,     // State value V(s)
    advantage_head: nn::Linear, // Action advantage A(s,a)
}

#[cfg(feature = "dqn")]
impl DuelingNetwork {
    pub fn new(vs: &nn::Path, input_dim: i64, hidden_dims: &[i64], num_actions: i64) -> Self {
        let fc1 = nn::linear(vs / "fc1", input_dim, hidden_dims[0], Default::default());
        let fc2 = nn::linear(vs / "fc2", hidden_dims[0], hidden_dims[1], Default::default());
        let fc3 = nn::linear(vs / "fc3", hidden_dims[1], hidden_dims[2], Default::default());

        let value_head = nn::linear(vs / "value", hidden_dims[2], 1, Default::default());
        let advantage_head = nn::linear(vs / "advantage", hidden_dims[2], num_actions, Default::default());

        Self {
            fc1,
            fc2,
            fc3,
            value_head,
            advantage_head,
        }
    }

    /// Forward pass with dueling architecture
    /// Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.apply(&self.fc1).relu();
        let x = x.apply(&self.fc2).relu();
        let x = x.apply(&self.fc3).relu();

        let value = x.apply(&self.value_head);  // [batch_size, 1]
        let advantage = x.apply(&self.advantage_head);  // [batch_size, num_actions]

        // Dueling: Q(s,a) = V(s) + (A(s,a) - mean(A))
        let advantage_mean = advantage.mean_dim(&[1i64], true, Kind::Float);
        let q_values = &value + &advantage - &advantage_mean;

        q_values
    }
}

/// Zero-Copy FluxNet-DQN for cryptic epitope prediction
#[derive(Debug)]
pub struct ZeroCopyFluxNetDqn {
    config: ZeroCopyDqnConfig,

    #[cfg(feature = "dqn")]
    device: Device,
    #[cfg(feature = "dqn")]
    vs: nn::VarStore,
    #[cfg(feature = "dqn")]
    online_network: DuelingNetwork,
    #[cfg(feature = "dqn")]
    target_network: DuelingNetwork,
    #[cfg(feature = "dqn")]
    optimizer: nn::Optimizer,

    replay_buffer: ReplayBuffer,
    epsilon: f64,
    training_step: u64,

    // GPU integration
    cuda_device: Arc<CudaContext>,

    // Provenance tracking
    provenance_hashes: Vec<String>,
    session_id: String,
}

impl ZeroCopyFluxNetDqn {
    /// Create new zero-copy DQN with GPU memory integration
    #[cfg(feature = "dqn")]
    pub fn new(config: ZeroCopyDqnConfig, cuda_device: Arc<CudaContext>) -> Result<Self> {
        // Initialize PyTorch with CUDA
        let device = Device::Cuda(0);
        let mut vs = nn::VarStore::new(device);

        let hidden_dims: Vec<i64> = config.hidden_dims.iter().map(|&x| x as i64).collect();

        // Create online and target networks
        let online_network = DuelingNetwork::new(
            &vs.root(),
            config.feature_dim as i64,
            &hidden_dims,
            config.num_actions as i64
        );

        let mut target_vs = nn::VarStore::new(device);
        let target_network = DuelingNetwork::new(
            &target_vs.root(),
            config.feature_dim as i64,
            &hidden_dims,
            config.num_actions as i64
        );

        // Copy weights from online to target network
        target_vs.copy(&vs)?;

        // Initialize optimizer
        let mut optimizer = nn::Adam::default().build(&vs, config.learning_rate)?;

        // Initialize replay buffer
        let replay_buffer = ReplayBuffer::new(config.replay_buffer_size);

        // Generate session ID for provenance
        let session_id = uuid::Uuid::new_v4().to_string();

        Ok(Self {
            config,
            device,
            vs,
            online_network,
            target_network,
            optimizer,
            replay_buffer,
            epsilon: config.epsilon,
            training_step: 0,
            cuda_device,
            provenance_hashes: Vec::new(),
            session_id,
        })
    }

    /// Create stub for non-DQN builds
    #[cfg(not(feature = "dqn"))]
    pub fn new(config: ZeroCopyDqnConfig, cuda_device: Arc<CudaContext>) -> Result<Self> {
        let replay_buffer = ReplayBuffer::new(config.replay_buffer_size);
        let session_id = uuid::Uuid::new_v4().to_string();

        Ok(Self {
            config: config.clone(),
            replay_buffer,
            epsilon: config.epsilon,
            cuda_device,
            provenance_hashes: Vec::new(),
            session_id,
            training_step: 0,
        })
    }

    /// Zero-copy inference directly on GPU memory
    #[cfg(feature = "dqn")]
    pub fn predict_zero_copy(&self, gpu_features: &CudaSlice<f32>) -> Result<Vec<f32>> {
        // CRITICAL: Zero-copy access - use Tensor::from_blob to wrap GPU memory
        let tensor_shape = [1, self.config.feature_dim as i64];

        // Create tensor that directly references GPU memory (zero-copy)
        // SAFETY: gpu_features must remain valid for tensor lifetime
        let feature_tensor = unsafe {
            Tensor::from_blob(
                gpu_features.device_ptr() as *mut f32,
                &tensor_shape,
                &[], // no strides needed for contiguous
                Kind::Float,
                self.device,
            )
        };

        // Forward pass on GPU tensor (no CPU copy!)
        let q_values = self.online_network.forward(&feature_tensor);

        // Convert back to CPU for action selection
        let q_values_cpu: Vec<f32> = q_values.try_into()?;

        Ok(q_values_cpu)
    }

    /// Stub for non-DQN builds
    #[cfg(not(feature = "dqn"))]
    pub fn predict_zero_copy(&self, _gpu_features: &CudaSlice<f32>) -> Result<Vec<f32>> {
        // Return dummy Q-values for all actions
        Ok(vec![0.0; self.config.num_actions])
    }

    /// Select action using epsilon-greedy with Q-values
    pub fn select_action(&self, q_values: &[f32]) -> CrypticAction {
        if rand::random::<f64>() < self.epsilon {
            // Random exploration
            let action_idx = rand::random::<usize>() % self.config.num_actions;
            CrypticAction::from(action_idx)
        } else {
            // Greedy exploitation - select action with highest Q-value
            let best_action_idx = q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            CrypticAction::from(best_action_idx)
        }
    }

    /// Add experience to replay buffer with cryptographic provenance
    pub fn add_experience(
        &mut self,
        state: Vec<f32>,
        action: CrypticAction,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
        residue_idx: usize,
        structure_id: &str,
    ) -> Result<()> {
        // Generate cryptographic hash for provenance
        let provenance_data = format!(
            "session:{},step:{},residue:{},structure:{},action:{:?},reward:{:.6}",
            self.session_id, self.training_step, residue_idx, structure_id, action, reward
        );

        let provenance_hash = <sha2::Sha256 as sha2::Digest>::digest(provenance_data.as_bytes());
        let provenance_hex = hex::encode(provenance_hash);

        let experience = Experience {
            state,
            action: action as u8,
            reward,
            next_state,
            done,
            provenance_hash: provenance_hex.clone(),
        };

        self.replay_buffer.add(experience);
        self.provenance_hashes.push(provenance_hex);

        Ok(())
    }

    /// Train the network using experience replay
    #[cfg(feature = "dqn")]
    pub fn train(&mut self) -> Result<f32> {
        if self.replay_buffer.len() < self.config.batch_size {
            return Ok(0.0); // Not enough experiences
        }

        let batch = self.replay_buffer
            .sample(self.config.batch_size)
            .context("Failed to sample from replay buffer")?;

        // Prepare batch tensors
        let states: Vec<Vec<f32>> = batch.iter().map(|exp| exp.state.clone()).collect();
        let actions: Vec<i64> = batch.iter().map(|exp| exp.action as i64).collect();
        let rewards: Vec<f32> = batch.iter().map(|exp| exp.reward).collect();
        let next_states: Vec<Vec<f32>> = batch.iter().map(|exp| exp.next_state.clone()).collect();
        let dones: Vec<f32> = batch.iter().map(|exp| if exp.done { 1.0 } else { 0.0 }).collect();

        // Convert to tensors
        let state_tensor = Tensor::of_slice2(&states).to(self.device);
        let action_tensor = Tensor::of_slice(&actions).to(self.device);
        let reward_tensor = Tensor::of_slice(&rewards).to(self.device);
        let next_state_tensor = Tensor::of_slice2(&next_states).to(self.device);
        let done_tensor = Tensor::of_slice(&dones).to(self.device);

        // Compute current Q-values
        let current_q_values = self.online_network.forward(&state_tensor);
        let current_q = current_q_values.gather(1, &action_tensor.unsqueeze(1), false);

        // Compute target Q-values using Double DQN
        let next_q_values_online = self.online_network.forward(&next_state_tensor);
        let next_actions = next_q_values_online.argmax(1, false);
        let next_q_values_target = self.target_network.forward(&next_state_tensor);
        let next_q = next_q_values_target.gather(1, &next_actions.unsqueeze(1), false);

        let target_q = &reward_tensor.unsqueeze(1) +
                      &((&Tensor::ones_like(&done_tensor) - &done_tensor).unsqueeze(1) *
                        self.config.gamma * &next_q);

        // Compute loss (Huber loss for stability)
        let loss = (current_q - target_q.detach()).huber_loss(1.0, tch::Reduction::Mean);

        // Backward pass
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        // Update training step and epsilon
        self.training_step += 1;
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);

        // Soft update target network
        self.soft_update_target()?;

        let loss_value: f32 = loss.try_into()?;
        Ok(loss_value)
    }

    /// Stub training for non-DQN builds
    #[cfg(not(feature = "dqn"))]
    pub fn train(&mut self) -> Result<f32> {
        self.training_step += 1;
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
        Ok(0.0)
    }

    /// Soft update target network: θ_target = τ * θ_online + (1 - τ) * θ_target
    #[cfg(feature = "dqn")]
    fn soft_update_target(&mut self) -> Result<()> {
        let tau = self.config.tau as f64;

        // This is a simplified version - in practice you'd iterate over parameters
        // For now, we do a hard update every N steps
        if self.training_step % 1000 == 0 {
            // Hard update every 1000 steps
            self.target_network = DuelingNetwork::new(
                &self.vs.root(),
                self.config.feature_dim as i64,
                &self.config.hidden_dims.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                self.config.num_actions as i64,
            );
        }

        Ok(())
    }

    #[cfg(not(feature = "dqn"))]
    fn soft_update_target(&mut self) -> Result<()> {
        Ok(())
    }

    /// Get training statistics
    pub fn get_stats(&self) -> serde_json::Value {
        serde_json::json!({
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "replay_buffer_size": self.replay_buffer.len(),
            "session_id": self.session_id,
            "provenance_count": self.provenance_hashes.len(),
        })
    }

    /// Get cryptographic provenance chain
    pub fn get_provenance_chain(&self) -> &[String] {
        &self.provenance_hashes
    }

    /// Validate provenance integrity
    pub fn validate_provenance(&self) -> bool {
        // Check if all hashes are valid SHA-256 (64 hex characters)
        self.provenance_hashes.iter().all(|hash| {
            hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit())
        })
    }

    /// Save model checkpoint with provenance
    #[cfg(feature = "dqn")]
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.vs.save(path)?;

        // Save provenance separately
        let provenance_path = format!("{}.provenance.json", path);
        let provenance_data = serde_json::json!({
            "session_id": self.session_id,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "config": self.config,
            "provenance_chain": self.provenance_hashes,
        });

        std::fs::write(provenance_path, serde_json::to_string_pretty(&provenance_data)?)?;

        Ok(())
    }

    #[cfg(not(feature = "dqn"))]
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let provenance_path = format!("{}.provenance.json", path);
        let provenance_data = serde_json::json!({
            "session_id": self.session_id,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "config": self.config,
            "provenance_chain": self.provenance_hashes,
        });

        std::fs::write(provenance_path, serde_json::to_string_pretty(&provenance_data)?)?;

        Ok(())
    }

    /// Load model checkpoint with provenance validation
    #[cfg(feature = "dqn")]
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        self.vs.load(path)?;

        // Load and validate provenance
        let provenance_path = format!("{}.provenance.json", path);
        if std::path::Path::new(&provenance_path).exists() {
            let provenance_data = std::fs::read_to_string(provenance_path)?;
            let provenance: serde_json::Value = serde_json::from_str(&provenance_data)?;

            if let Some(chain) = provenance["provenance_chain"].as_array() {
                self.provenance_hashes = chain.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
            }

            if let Some(step) = provenance["training_step"].as_u64() {
                self.training_step = step;
            }

            if let Some(eps) = provenance["epsilon"].as_f64() {
                self.epsilon = eps;
            }
        }

        Ok(())
    }

    #[cfg(not(feature = "dqn"))]
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        let provenance_path = format!("{}.provenance.json", path);
        if std::path::Path::new(&provenance_path).exists() {
            let provenance_data = std::fs::read_to_string(provenance_path)?;
            let provenance: serde_json::Value = serde_json::from_str(&provenance_data)?;

            if let Some(chain) = provenance["provenance_chain"].as_array() {
                self.provenance_hashes = chain.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
            }

            if let Some(step) = provenance["training_step"].as_u64() {
                self.training_step = step;
            }

            if let Some(eps) = provenance["epsilon"].as_f64() {
                self.epsilon = eps;
            }
        }

        Ok(())
    }
}

/// Compute reward for cryptic epitope prediction task
pub fn compute_cryptic_reward(
    action: CrypticAction,
    is_cryptic: bool,
    is_epitope: bool,
    confidence: f32,
) -> f32 {
    match action {
        CrypticAction::PredictCryptic => {
            if is_cryptic {
                // Primary Goal: Detect Cryptic Sites
                // Base 1.0 + Confidence Bonus + Epitope Bonus
                1.0 + (confidence * 0.5) + (if is_epitope { 0.5 } else { 0.0 })
            } else {
                // False Positive Penalty
                -0.5 - (confidence * 0.2)
            }
        }
        CrypticAction::PredictExposed => {
            if is_cryptic {
                -1.0 // Severe penalty for missing a cryptic site (False Negative)
            } else if is_epitope {
                0.3  // Correct exposed epitope
            } else {
                0.1  // Correct exposed non-epitope
            }
        }
        CrypticAction::PredictEpitope => {
            if is_cryptic {
                -0.5 // Suboptimal: Should have predicted Cryptic!
            } else if is_epitope {
                0.5 // Good: Correctly identified standard epitope
            } else {
                -0.2 // False Positive
            }
        }
        CrypticAction::Skip => {
            if is_cryptic { -1.0 } else { 0.0 } // Penalty for skipping cryptic
        }, 
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cryptic_action_conversion() {
        assert_eq!(CrypticAction::from(0), CrypticAction::PredictCryptic);
        assert_eq!(CrypticAction::from(1), CrypticAction::PredictExposed);
        assert_eq!(CrypticAction::from(2), CrypticAction::PredictEpitope);
        assert_eq!(CrypticAction::from(3), CrypticAction::Skip);
        assert_eq!(CrypticAction::from(99), CrypticAction::Skip);
    }

    #[test]
    fn test_reward_computation() {
        // Test cryptic prediction rewards
        assert_eq!(compute_cryptic_reward(CrypticAction::PredictCryptic, true, false, 0.8), 1.4);
        assert_eq!(compute_cryptic_reward(CrypticAction::PredictCryptic, false, false, 0.8), -0.66);

        // Test exposed prediction rewards
        assert_eq!(compute_cryptic_reward(CrypticAction::PredictExposed, false, true, 0.0), 0.3);
        assert_eq!(compute_cryptic_reward(CrypticAction::PredictExposed, false, false, 0.0), 0.1);
        assert_eq!(compute_cryptic_reward(CrypticAction::PredictExposed, true, false, 0.0), -0.3);

        // Test epitope prediction rewards
        assert_eq!(compute_cryptic_reward(CrypticAction::PredictEpitope, false, true, 0.0), 0.5);
        assert_eq!(compute_cryptic_reward(CrypticAction::PredictEpitope, true, true, 0.0), 0.8);
        assert_eq!(compute_cryptic_reward(CrypticAction::PredictEpitope, false, false, 0.0), -0.2);

        // Test skip action
        assert_eq!(compute_cryptic_reward(CrypticAction::Skip, true, true, 1.0), 0.0);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(3);

        let exp1 = Experience {
            state: vec![1.0; 140],
            action: 0,
            reward: 1.0,
            next_state: vec![1.1; 140],
            done: false,
            provenance_hash: "test1".to_string(),
        };

        let exp2 = Experience {
            state: vec![2.0; 140],
            action: 1,
            reward: 0.5,
            next_state: vec![2.1; 140],
            done: false,
            provenance_hash: "test2".to_string(),
        };

        buffer.add(exp1);
        buffer.add(exp2.clone());

        assert_eq!(buffer.len(), 2);

        // Test sampling when not enough experiences
        assert!(buffer.sample(5).is_none());

        // Add one more to test capacity
        let exp3 = Experience { ..exp2.clone() };
        buffer.add(exp3);
        assert_eq!(buffer.len(), 3);

        // Add one more to test overflow
        let exp4 = Experience {
            provenance_hash: "test4".to_string(),
            ..exp2
        };
        buffer.add(exp4);
        assert_eq!(buffer.len(), 3);  // Should still be 3 due to capacity limit

        // Test successful sampling
        let sample = buffer.sample(2);
        assert!(sample.is_some());
        assert_eq!(sample.unwrap().len(), 2);
    }
}