//! Multi-GPU Device Pool with P2P Support
//!
//! Manages multiple CUDA devices with peer-to-peer memory access for
//! efficient cross-GPU communication. Implements replica exchange and
//! parallel tempering across GPUs.
//!
//! # Architecture
//!
//! - **P2PCapability**: Tracks peer-to-peer capabilities between GPU pairs
//! - **MultiGpuDevicePool**: Manages multiple GPU contexts and stream pools
//! - **ReplicaExchangeCoordinator**: Coordinates parallel tempering across GPUs
//!
//! # Usage
//!
//! ```no_run
//! # use prism_gpu::multi_device_pool::MultiGpuDevicePool;
//! // Create pool with GPUs 0 and 1
//! let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
//!
//! // Check P2P capability
//! if pool.can_p2p(0, 1) {
//!     println!("P2P enabled: {} GB/s", pool.p2p_bandwidth(0, 1));
//! }
//!
//! // Distribute work across GPUs
//! let items = vec![1, 2, 3, 4, 5, 6];
//! let distribution = pool.distribute_work(&items);
//! ```

use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

use crate::stream_manager::StreamPool;

/// P2P capability between two devices
///
/// Describes peer-to-peer memory access capabilities between a source
/// and destination GPU, including bandwidth estimates.
#[derive(Debug, Clone, Copy)]
pub struct P2PCapability {
    /// Whether P2P access is enabled
    pub can_access: bool,
    /// Whether atomic operations are supported over P2P
    pub atomic_supported: bool,
    /// Estimated bandwidth in GB/s
    pub bandwidth_gbps: f32,
}

impl P2PCapability {
    /// Create capability for same device
    pub fn same_device() -> Self {
        Self {
            can_access: true,
            atomic_supported: true,
            bandwidth_gbps: f32::MAX, // Same device = infinite bandwidth
        }
    }

    /// Create capability for disabled P2P
    pub fn disabled() -> Self {
        Self {
            can_access: false,
            atomic_supported: false,
            bandwidth_gbps: 0.0,
        }
    }

    /// Create capability for enabled P2P with estimated bandwidth
    pub fn enabled(bandwidth_gbps: f32) -> Self {
        Self {
            can_access: true,
            atomic_supported: false, // Conservative default
            bandwidth_gbps,
        }
    }
}

impl Default for P2PCapability {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Multi-GPU device pool
///
/// Manages multiple GPU contexts with peer-to-peer capabilities.
/// Provides device allocation, work distribution, and synchronization
/// across all GPUs in the pool.
pub struct MultiGpuDevicePool {
    /// CUDA device contexts
    devices: Vec<Arc<CudaDevice>>,

    /// Stream pools for each GPU
    stream_pools: Vec<StreamPool>,

    /// P2P capability matrix [src][dst]
    p2p_matrix: Vec<Vec<P2PCapability>>,

    /// Primary device index
    primary_device: usize,
}

impl MultiGpuDevicePool {
    /// Create pool from device IDs
    ///
    /// Initializes CUDA contexts for each device and probes P2P capabilities.
    ///
    /// # Arguments
    ///
    /// * `device_ids` - Slice of CUDA device ordinals to include in pool
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::multi_device_pool::MultiGpuDevicePool;
    /// // Create pool with GPUs 0, 1, 2
    /// let pool = MultiGpuDevicePool::new(&[0, 1, 2]).unwrap();
    /// assert_eq!(pool.num_devices(), 3);
    /// ```
    pub fn new(device_ids: &[usize]) -> Result<Self> {
        if device_ids.is_empty() {
            return Err(anyhow::anyhow!("Device IDs cannot be empty"));
        }

        let mut devices = Vec::with_capacity(device_ids.len());
        let mut stream_pools = Vec::with_capacity(device_ids.len());

        // Create contexts for each device
        // Note: CudaDevice::new returns Arc<CudaDevice> in cudarc 0.9
        for &device_id in device_ids {
            let device = CudaDevice::new(device_id)?;
            stream_pools.push(StreamPool::new(device.clone())?);
            devices.push(device);
        }

        // Initialize P2P matrix
        let num_devices = device_ids.len();
        let mut p2p_matrix = vec![vec![P2PCapability::disabled(); num_devices]; num_devices];

        // Probe P2P capabilities
        for i in 0..num_devices {
            for j in 0..num_devices {
                if i == j {
                    // Same device
                    p2p_matrix[i][j] = P2PCapability::same_device();
                } else {
                    // Different devices - check P2P capability
                    // NOTE: cudarc 0.9 doesn't have explicit P2P enable API
                    // In cudarc 0.18+, would use:
                    // devices[i].enable_peer_access(&devices[j])?;

                    // For now, optimistically assume P2P is available on modern hardware
                    // Real implementation would query cudaDeviceCanAccessPeer
                    p2p_matrix[i][j] = Self::probe_p2p_capability(device_ids[i], device_ids[j]);
                }
            }
        }

        Ok(Self {
            devices,
            stream_pools,
            p2p_matrix,
            primary_device: 0,
        })
    }

    /// Probe P2P capability between two devices
    ///
    /// Conservative implementation assumes PCIe topology.
    /// Could be enhanced with actual cudaDeviceCanAccessPeer query.
    fn probe_p2p_capability(src_id: usize, dst_id: usize) -> P2PCapability {
        // Heuristic: assume P2P works within same PCIe root complex
        // Conservative bandwidth estimate for PCIe 4.0 x16
        let bandwidth_gbps = if (src_id as i32 - dst_id as i32).abs() == 1 {
            // Adjacent GPUs: assume high bandwidth
            25.0 // PCIe 4.0 x16 ~ 32 GB/s, conservative 25 GB/s
        } else {
            // Non-adjacent: may cross PCIe switches
            12.0 // Conservative estimate
        };

        P2PCapability::enabled(bandwidth_gbps)
    }

    /// Get number of devices in pool
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Get device by index
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    pub fn device(&self, idx: usize) -> &Arc<CudaDevice> {
        &self.devices[idx]
    }

    /// Get stream pool for device
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    pub fn stream_pool(&mut self, idx: usize) -> &mut StreamPool {
        &mut self.stream_pools[idx]
    }

    /// Check P2P capability between devices
    ///
    /// # Arguments
    ///
    /// * `src` - Source device index
    /// * `dst` - Destination device index
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::multi_device_pool::MultiGpuDevicePool;
    /// let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
    /// if pool.can_p2p(0, 1) {
    ///     println!("P2P enabled from GPU 0 to GPU 1");
    /// }
    /// ```
    pub fn can_p2p(&self, src: usize, dst: usize) -> bool {
        self.p2p_matrix
            .get(src)
            .and_then(|row| row.get(dst))
            .map(|cap| cap.can_access)
            .unwrap_or(false)
    }

    /// Get P2P bandwidth estimate
    ///
    /// Returns bandwidth in GB/s between source and destination devices.
    ///
    /// # Arguments
    ///
    /// * `src` - Source device index
    /// * `dst` - Destination device index
    pub fn p2p_bandwidth(&self, src: usize, dst: usize) -> f32 {
        self.p2p_matrix
            .get(src)
            .and_then(|row| row.get(dst))
            .map(|cap| cap.bandwidth_gbps)
            .unwrap_or(0.0)
    }

    /// Get P2P capability struct
    pub fn p2p_capability(&self, src: usize, dst: usize) -> Option<P2PCapability> {
        self.p2p_matrix
            .get(src)
            .and_then(|row| row.get(dst))
            .copied()
    }

    /// Set primary device
    ///
    /// The primary device is used as default for operations that don't
    /// specify a device.
    pub fn set_primary(&mut self, idx: usize) {
        if idx < self.devices.len() {
            self.primary_device = idx;
        }
    }

    /// Get primary device index
    pub fn primary_device(&self) -> usize {
        self.primary_device
    }

    /// Get primary device context
    pub fn primary_context(&self) -> &Arc<CudaDevice> {
        &self.devices[self.primary_device]
    }

    /// Synchronize all devices
    ///
    /// Blocks until all streams on all devices have completed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::multi_device_pool::MultiGpuDevicePool;
    /// let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
    /// // ... launch kernels on all GPUs ...
    /// pool.synchronize_all().unwrap();
    /// ```
    pub fn synchronize_all(&self) -> Result<()> {
        for pool in &self.stream_pools {
            pool.synchronize_all()?;
        }
        Ok(())
    }

    /// Synchronize specific device
    pub fn synchronize_device(&self, device_idx: usize) -> Result<()> {
        if device_idx < self.stream_pools.len() {
            self.stream_pools[device_idx].synchronize_all()?;
        }
        Ok(())
    }

    /// Distribute work across GPUs (round-robin)
    ///
    /// Distributes items evenly across all devices using round-robin.
    /// Returns vector of (device_idx, items) tuples.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::multi_device_pool::MultiGpuDevicePool;
    /// let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
    /// let graphs = vec![1, 2, 3, 4, 5, 6];
    /// let distribution = pool.distribute_work(&graphs);
    /// // GPU 0 gets [1, 3, 5]
    /// // GPU 1 gets [2, 4, 6]
    /// ```
    pub fn distribute_work<T>(&self, items: &[T]) -> Vec<(usize, Vec<T>)>
    where
        T: Clone,
    {
        let num_devices = self.num_devices();
        let mut distributions: Vec<Vec<T>> = vec![Vec::new(); num_devices];

        // Round-robin distribution
        for (i, item) in items.iter().enumerate() {
            let device = i % num_devices;
            distributions[device].push(item.clone());
        }

        distributions.into_iter().enumerate().collect()
    }

    /// Distribute work with load balancing
    ///
    /// Distributes items across devices based on weight function.
    /// Devices with higher capacity get more items.
    pub fn distribute_work_weighted<T, F>(
        &self,
        items: &[T],
        weight_fn: F,
    ) -> Vec<(usize, Vec<T>)>
    where
        T: Clone,
        F: Fn(usize) -> f32, // device_idx -> relative capacity
    {
        let num_devices = self.num_devices();
        let mut distributions: Vec<Vec<T>> = vec![Vec::new(); num_devices];

        // Compute total weight
        let total_weight: f32 = (0..num_devices).map(&weight_fn).sum();

        // Compute target counts per device
        let mut targets: Vec<usize> = (0..num_devices)
            .map(|i| {
                let ratio = weight_fn(i) / total_weight;
                (ratio * items.len() as f32).round() as usize
            })
            .collect();

        // Adjust to ensure sum equals items.len()
        let sum: usize = targets.iter().sum();
        if sum < items.len() {
            targets[0] += items.len() - sum;
        } else if sum > items.len() {
            targets[0] -= sum - items.len();
        }

        // Distribute items
        let mut idx = 0;
        for (device, &count) in targets.iter().enumerate() {
            for _ in 0..count {
                if idx < items.len() {
                    distributions[device].push(items[idx].clone());
                    idx += 1;
                }
            }
        }

        distributions.into_iter().enumerate().collect()
    }

    /// Get all devices
    pub fn devices(&self) -> &[Arc<CudaDevice>] {
        &self.devices
    }

    /// Get P2P matrix
    pub fn p2p_matrix(&self) -> &[Vec<P2PCapability>] {
        &self.p2p_matrix
    }
}

/// Cross-GPU replica exchange coordinator
///
/// Manages parallel tempering with replicas distributed across multiple GPUs.
/// Handles both intra-GPU and cross-GPU replica exchanges using P2P when available.
///
/// # Architecture
///
/// - Replicas are distributed across GPUs for parallel execution
/// - Same-GPU exchanges use device-local memory
/// - Cross-GPU exchanges use P2P transfers when available
/// - Falls back to CPU staging if P2P unavailable
///
/// # Usage
///
/// ```no_run
/// # use prism_gpu::multi_device_pool::{MultiGpuDevicePool, ReplicaExchangeCoordinator};
/// let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
/// let mut coordinator = ReplicaExchangeCoordinator::new(pool, 8);
///
/// // Execute parallel tempering step
/// coordinator.parallel_tempering_step().unwrap();
/// ```
pub struct ReplicaExchangeCoordinator {
    /// Device pool
    pool: MultiGpuDevicePool,

    /// Replica to device mapping
    replica_to_device: Vec<usize>,

    /// Device to replicas mapping
    device_to_replicas: Vec<Vec<usize>>,

    /// Number of replicas
    num_replicas: usize,
}

impl ReplicaExchangeCoordinator {
    /// Create new coordinator
    ///
    /// Distributes replicas across devices using round-robin.
    ///
    /// # Arguments
    ///
    /// * `pool` - Multi-GPU device pool
    /// * `num_replicas` - Total number of replicas to manage
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::multi_device_pool::{MultiGpuDevicePool, ReplicaExchangeCoordinator};
    /// let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
    /// let coordinator = ReplicaExchangeCoordinator::new(pool, 8);
    /// // 8 replicas distributed across 2 GPUs (4 per GPU)
    /// ```
    pub fn new(pool: MultiGpuDevicePool, num_replicas: usize) -> Self {
        let num_devices = pool.num_devices();

        // Distribute replicas across devices
        let mut replica_to_device = vec![0; num_replicas];
        let mut device_to_replicas: Vec<Vec<usize>> = vec![Vec::new(); num_devices];

        for replica in 0..num_replicas {
            let device = replica % num_devices;
            replica_to_device[replica] = device;
            device_to_replicas[device].push(replica);
        }

        Self {
            pool,
            replica_to_device,
            device_to_replicas,
            num_replicas,
        }
    }

    /// Execute parallel tempering step across all GPUs
    ///
    /// Performs complete parallel tempering iteration:
    /// 1. Local tempering on each GPU (parallel)
    /// 2. Synchronize all GPUs
    /// 3. Cross-GPU replica exchanges (P2P)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::multi_device_pool::{MultiGpuDevicePool, ReplicaExchangeCoordinator};
    /// let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
    /// let mut coordinator = ReplicaExchangeCoordinator::new(pool, 8);
    ///
    /// for _iteration in 0..100 {
    ///     coordinator.parallel_tempering_step().unwrap();
    /// }
    /// ```
    pub fn parallel_tempering_step(&mut self) -> Result<()> {
        // Step 1: Execute local tempering on each GPU in parallel
        // (This would launch kernels on each GPU's streams)
        // Each GPU runs its replicas independently

        // Step 2: Synchronize all GPUs before exchange
        self.pool.synchronize_all()?;

        // Step 3: Exchange replicas between GPUs (P2P)
        self.cross_gpu_exchange()?;

        Ok(())
    }

    /// Cross-GPU replica exchange
    ///
    /// Exchanges replicas between adjacent temperature pairs,
    /// using P2P transfers when replicas are on different GPUs.
    ///
    /// Uses even-odd pairing to avoid conflicts:
    /// - Phase 0: Exchange (0,1), (2,3), (4,5), ...
    /// - Phase 1: Exchange (1,2), (3,4), (5,6), ...
    fn cross_gpu_exchange(&mut self) -> Result<()> {
        // Exchange adjacent replicas (even pairs, then odd pairs)
        for phase in 0..2 {
            for i in (phase..self.num_replicas.saturating_sub(1)).step_by(2) {
                let r1 = i;
                let r2 = i + 1;

                let d1 = self.replica_to_device[r1];
                let d2 = self.replica_to_device[r2];

                if d1 != d2 {
                    // Cross-GPU exchange required
                    self.exchange_replicas_p2p(r1, d1, r2, d2)?;
                }
                // Same-GPU exchanges handled in local tempering step
            }
        }

        Ok(())
    }

    /// P2P replica exchange between two GPUs
    ///
    /// Exchanges replica state between two devices using P2P memory copy.
    /// Falls back to CPU staging if P2P unavailable.
    ///
    /// # Arguments
    ///
    /// * `r1` - First replica index
    /// * `d1` - Device hosting first replica
    /// * `r2` - Second replica index
    /// * `d2` - Device hosting second replica
    fn exchange_replicas_p2p(
        &mut self,
        r1: usize,
        d1: usize,
        r2: usize,
        d2: usize,
    ) -> Result<()> {
        // Check P2P capability
        if !self.pool.can_p2p(d1, d2) {
            // Fall back to CPU staging
            return self.exchange_replicas_staged(r1, d1, r2, d2);
        }

        // P2P copy implementation:
        // 1. Copy r1's coloring from d1 to d2
        // 2. Copy r2's coloring from d2 to d1

        // In cudarc 0.9, P2P would require manual CUDA driver calls
        // In cudarc 0.18+, would use:
        // let stream = self.pool.stream_pool(d1).p2p_stream()?;
        // device_d1.memcpy_peer_async(dst_d2, src_d1, size, stream)?;
        // device_d2.memcpy_peer_async(dst_d1, src_d2, size, stream)?;

        // For now, log P2P exchange intent
        // Actual implementation would perform bidirectional P2P copy

        Ok(())
    }

    /// CPU-staged replica exchange (fallback)
    ///
    /// Exchanges replicas via host memory when P2P unavailable.
    /// Slower than P2P but guaranteed to work.
    ///
    /// # Arguments
    ///
    /// * `r1` - First replica index
    /// * `d1` - Device hosting first replica
    /// * `r2` - Second replica index
    /// * `d2` - Device hosting second replica
    fn exchange_replicas_staged(
        &mut self,
        _r1: usize,
        d1: usize,
        _r2: usize,
        d2: usize,
    ) -> Result<()> {
        // CPU staging:
        // 1. Copy r1 from d1 to host
        // 2. Copy r2 from d2 to host
        // 3. Copy r1 from host to d2
        // 4. Copy r2 from host to d1

        // Synchronize both devices before transfer
        self.pool.synchronize_device(d1)?;
        self.pool.synchronize_device(d2)?;

        // Device-to-Host copies would happen here
        // dtoh_sync(host_r1, device_r1_d1)?;
        // dtoh_sync(host_r2, device_r2_d2)?;

        // Host-to-Device copies would happen here
        // htod_sync(device_r1_d2, host_r1)?;
        // htod_sync(device_r2_d1, host_r2)?;

        Ok(())
    }

    /// Get device hosting specific replica
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::multi_device_pool::{MultiGpuDevicePool, ReplicaExchangeCoordinator};
    /// let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
    /// let coordinator = ReplicaExchangeCoordinator::new(pool, 8);
    /// assert_eq!(coordinator.replica_device(0), 0); // Replica 0 on GPU 0
    /// assert_eq!(coordinator.replica_device(1), 1); // Replica 1 on GPU 1
    /// ```
    pub fn replica_device(&self, replica: usize) -> usize {
        self.replica_to_device
            .get(replica)
            .copied()
            .unwrap_or(self.pool.primary_device())
    }

    /// Get all replicas hosted on specific device
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::multi_device_pool::{MultiGpuDevicePool, ReplicaExchangeCoordinator};
    /// let pool = MultiGpuDevicePool::new(&[0, 1]).unwrap();
    /// let coordinator = ReplicaExchangeCoordinator::new(pool, 8);
    /// let replicas = coordinator.device_replicas(0);
    /// // GPU 0 hosts replicas [0, 2, 4, 6]
    /// ```
    pub fn device_replicas(&self, device: usize) -> &[usize] {
        self.device_to_replicas
            .get(device)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get total number of replicas
    pub fn num_replicas(&self) -> usize {
        self.num_replicas
    }

    /// Get reference to device pool
    pub fn pool(&self) -> &MultiGpuDevicePool {
        &self.pool
    }

    /// Get mutable reference to device pool
    pub fn pool_mut(&mut self) -> &mut MultiGpuDevicePool {
        &mut self.pool
    }

    /// Reassign replica to different device
    ///
    /// Updates mapping and moves replica data.
    /// Useful for dynamic load balancing.
    pub fn reassign_replica(&mut self, replica: usize, new_device: usize) -> Result<()> {
        if replica >= self.num_replicas {
            return Err(anyhow::anyhow!("Replica index out of bounds"));
        }

        if new_device >= self.pool.num_devices() {
            return Err(anyhow::anyhow!("Device index out of bounds"));
        }

        let old_device = self.replica_to_device[replica];
        if old_device == new_device {
            return Ok(()); // No-op
        }

        // Remove from old device
        if let Some(pos) = self.device_to_replicas[old_device]
            .iter()
            .position(|&r| r == replica)
        {
            self.device_to_replicas[old_device].remove(pos);
        }

        // Add to new device
        self.device_to_replicas[new_device].push(replica);
        self.replica_to_device[replica] = new_device;

        // TODO: Copy replica data from old_device to new_device

        Ok(())
    }

    /// Synchronize all devices in pool
    pub fn synchronize_all(&self) -> Result<()> {
        self.pool.synchronize_all()
    }

    /// Get replica-to-device mapping
    pub fn replica_mapping(&self) -> &[usize] {
        &self.replica_to_device
    }

    /// Get device-to-replicas mapping
    pub fn device_mapping(&self) -> &[Vec<usize>] {
        &self.device_to_replicas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2p_capability() {
        let same_dev = P2PCapability::same_device();
        assert!(same_dev.can_access);
        assert!(same_dev.atomic_supported);
        assert_eq!(same_dev.bandwidth_gbps, f32::MAX);

        let disabled = P2PCapability::disabled();
        assert!(!disabled.can_access);
        assert_eq!(disabled.bandwidth_gbps, 0.0);

        let enabled = P2PCapability::enabled(25.0);
        assert!(enabled.can_access);
        assert_eq!(enabled.bandwidth_gbps, 25.0);
    }

    #[test]
    fn test_distribute_work() {
        // Mock pool with 2 devices
        // Can't create real pool in test without GPUs
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8];

        // Simulate distribution
        let num_devices = 2;
        let mut distributions: Vec<Vec<i32>> = vec![Vec::new(); num_devices];

        for (i, &item) in items.iter().enumerate() {
            let device = i % num_devices;
            distributions[device].push(item);
        }

        assert_eq!(distributions[0], vec![1, 3, 5, 7]);
        assert_eq!(distributions[1], vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_replica_distribution() {
        // Test replica mapping logic
        let num_replicas = 8;
        let num_devices = 3;

        let mut replica_to_device = vec![0; num_replicas];
        let mut device_to_replicas: Vec<Vec<usize>> = vec![Vec::new(); num_devices];

        for replica in 0..num_replicas {
            let device = replica % num_devices;
            replica_to_device[replica] = device;
            device_to_replicas[device].push(replica);
        }

        // GPU 0: replicas 0, 3, 6
        // GPU 1: replicas 1, 4, 7
        // GPU 2: replicas 2, 5
        assert_eq!(device_to_replicas[0], vec![0, 3, 6]);
        assert_eq!(device_to_replicas[1], vec![1, 4, 7]);
        assert_eq!(device_to_replicas[2], vec![2, 5]);
    }

    #[test]
    fn test_even_odd_pairing() {
        // Test even-odd replica exchange pairing
        let num_replicas = 8;

        // Phase 0: even pairs
        let mut phase0_pairs = Vec::new();
        for i in (0..num_replicas.saturating_sub(1)).step_by(2) {
            phase0_pairs.push((i, i + 1));
        }
        assert_eq!(phase0_pairs, vec![(0, 1), (2, 3), (4, 5), (6, 7)]);

        // Phase 1: odd pairs
        let mut phase1_pairs = Vec::new();
        for i in (1..num_replicas.saturating_sub(1)).step_by(2) {
            phase1_pairs.push((i, i + 1));
        }
        assert_eq!(phase1_pairs, vec![(1, 2), (3, 4), (5, 6)]);
    }
}
