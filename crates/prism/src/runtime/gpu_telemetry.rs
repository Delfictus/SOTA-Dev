//! GPU Telemetry - Real-time GPU Monitoring
//!
//! Polls GPU status and emits events for UI updates and metrics.
//! Supports multiple integration strategies:
//! 1. NVML (NVIDIA Management Library) - production
//! 2. cudarc device queries - fallback
//! 3. Simulated data - development/testing

use super::events::*;
use super::state::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use anyhow::Result;

/// GPU Telemetry collector
pub struct GpuTelemetry {
    /// Device ID to monitor
    device_id: usize,
    /// Polling interval in milliseconds
    poll_interval_ms: u64,
    /// Strategy for collecting GPU data
    strategy: TelemetryStrategy,
}

/// Strategy for collecting GPU telemetry
enum TelemetryStrategy {
    /// Use NVML for real data (TODO: integrate nvml-wrapper)
    #[allow(dead_code)]
    Nvml,
    /// Use cudarc device info (TODO: integrate cudarc queries)
    #[allow(dead_code)]
    Cudarc,
    /// Simulated data for development
    Simulated(SimulatedGpu),
}

/// Simulated GPU for development/testing
struct SimulatedGpu {
    name: String,
    memory_total: u64,
    base_utilization: f64,
    base_temperature: u32,
    variation_phase: f64,
}

impl SimulatedGpu {
    fn new(device_id: usize) -> Self {
        Self {
            name: format!("NVIDIA RTX 4090 (Simulated #{device_id})"),
            memory_total: 24 * 1024 * 1024 * 1024, // 24 GB
            base_utilization: 0.65,
            base_temperature: 55,
            variation_phase: 0.0,
        }
    }

    fn poll(&mut self) -> GpuSnapshot {
        // Simulate realistic GPU behavior with time-varying metrics
        self.variation_phase += 0.1;

        let utilization_variance = (self.variation_phase.sin() * 0.15).max(-0.1);
        let temp_variance = (self.variation_phase.cos() * 5.0) as i32;
        let memory_variance = ((self.variation_phase * 1.5).sin() * 0.1).max(-0.05);

        let utilization = (self.base_utilization + utilization_variance).clamp(0.0, 1.0);
        let temperature = (self.base_temperature as i32 + temp_variance).max(30) as u32;
        let memory_used_pct = (0.45 + memory_variance).clamp(0.1, 0.95);
        let memory_used = (self.memory_total as f64 * memory_used_pct) as u64;

        // Power correlates with utilization
        let power_watts = 150.0 + (utilization * 300.0);

        GpuSnapshot {
            name: self.name.clone(),
            utilization,
            memory_used,
            memory_total: self.memory_total,
            temperature,
            power_watts: power_watts as f32,
        }
    }
}

/// GPU status snapshot
struct GpuSnapshot {
    name: String,
    utilization: f64,
    memory_used: u64,
    memory_total: u64,
    temperature: u32,
    power_watts: f32,
}

impl GpuTelemetry {
    /// Create a new GPU telemetry collector
    pub fn new(device_id: usize, poll_interval_ms: u64) -> Self {
        // For now, use simulated strategy
        // TODO: Try NVML first, fall back to cudarc, then simulated
        let strategy = TelemetryStrategy::Simulated(SimulatedGpu::new(device_id));

        Self {
            device_id,
            poll_interval_ms,
            strategy,
        }
    }

    /// Poll current GPU status
    pub fn poll(&mut self) -> GpuSnapshot {
        match &mut self.strategy {
            TelemetryStrategy::Simulated(sim) => sim.poll(),
            // TODO: Implement NVML and cudarc strategies
            _ => unreachable!("NVML and cudarc strategies not yet implemented"),
        }
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Get poll interval
    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.poll_interval_ms)
    }
}

/// Run the GPU telemetry polling loop
///
/// This async task polls GPU status at regular intervals and:
/// 1. Updates the state store with current GPU metrics
/// 2. Emits GpuStatus events for reactive UI updates
/// 3. Records utilization history for charting
///
/// # Arguments
/// * `telemetry` - GPU telemetry collector
/// * `state` - Shared state store
/// * `event_bus` - Event bus for broadcasting updates
/// * `shutdown` - Shutdown signal receiver
pub async fn run_telemetry_loop(
    mut telemetry: GpuTelemetry,
    state: Arc<StateStore>,
    event_bus: Arc<EventBus>,
    mut shutdown: oneshot::Receiver<()>,
) -> Result<()> {
    let device_id = telemetry.device_id();
    let poll_interval = telemetry.poll_interval();

    log::info!(
        "GPU telemetry loop starting for device {} (poll interval: {:?})",
        device_id,
        poll_interval
    );

    let mut interval = tokio::time::interval(poll_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Poll GPU status
                let snapshot = telemetry.poll();

                // Update state store
                {
                    let mut gpu_states = state.gpu.write();
                    if device_id < gpu_states.len() {
                        let gpu_state = &mut gpu_states[device_id];
                        gpu_state.device_id = device_id;
                        gpu_state.name = snapshot.name.clone();
                        gpu_state.utilization = snapshot.utilization;
                        gpu_state.memory_used = snapshot.memory_used;
                        gpu_state.memory_total = snapshot.memory_total;
                        gpu_state.temperature = snapshot.temperature;
                        gpu_state.power_watts = snapshot.power_watts;
                    }
                }

                // Record utilization history
                let memory_pct = (snapshot.memory_used as f64 / snapshot.memory_total as f64) * 100.0;
                state.record_gpu_util(device_id, snapshot.utilization * 100.0, memory_pct);

                // Emit GPU status event
                let event = PrismEvent::GpuStatus {
                    device_id,
                    name: snapshot.name,
                    utilization: snapshot.utilization,
                    memory_used: snapshot.memory_used,
                    memory_total: snapshot.memory_total,
                    temperature: snapshot.temperature,
                    power_watts: snapshot.power_watts,
                };

                if let Err(e) = event_bus.publish(event).await {
                    log::warn!("Failed to publish GPU status event: {}", e);
                }
            }

            _ = &mut shutdown => {
                log::info!("GPU telemetry loop shutting down for device {}", device_id);
                break;
            }
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Future Integration Points
// ═══════════════════════════════════════════════════════════════════════════

/// Create telemetry with NVML if available
///
/// TODO: Implement when nvml-wrapper is added to dependencies
/// ```toml
/// [dependencies]
/// nvml-wrapper = { version = "0.9", optional = true }
/// ```
#[allow(dead_code)]
fn try_create_nvml_telemetry(_device_id: usize, _poll_interval_ms: u64) -> Option<GpuTelemetry> {
    // Example implementation:
    // use nvml_wrapper::Nvml;
    //
    // let nvml = Nvml::init().ok()?;
    // let device = nvml.device_by_index(device_id as u32).ok()?;
    //
    // Some(GpuTelemetry {
    //     device_id,
    //     poll_interval_ms,
    //     strategy: TelemetryStrategy::Nvml(NvmlDevice { device }),
    // })
    None
}

/// Create telemetry with cudarc device queries
///
/// TODO: Implement when cudarc device info API is integrated
#[allow(dead_code)]
fn try_create_cudarc_telemetry(_device_id: usize, _poll_interval_ms: u64) -> Option<GpuTelemetry> {
    // Example implementation:
    // use cudarc::driver::CudaDevice;
    //
    // let device = CudaDevice::new(device_id).ok()?;
    // let name = device.name().ok()?;
    // let total_mem = device.total_memory().ok()?;
    //
    // Some(GpuTelemetry {
    //     device_id,
    //     poll_interval_ms,
    //     strategy: TelemetryStrategy::Cudarc(CudarcDevice { device }),
    // })
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_gpu() {
        let mut sim = SimulatedGpu::new(0);

        // Poll multiple times to see variation
        let snapshot1 = sim.poll();
        assert!(snapshot1.utilization > 0.0 && snapshot1.utilization <= 1.0);
        assert!(snapshot1.temperature > 0);
        assert!(snapshot1.memory_used > 0);
        assert!(snapshot1.power_watts > 0.0);

        // Second poll should have different values
        let snapshot2 = sim.poll();
        // Due to sine wave simulation, values should vary
        assert_ne!(snapshot1.utilization, snapshot2.utilization);
    }

    #[tokio::test]
    async fn test_telemetry_loop() {
        let telemetry = GpuTelemetry::new(0, 50);
        let state = Arc::new(StateStore::new(100));
        let event_bus = Arc::new(EventBus::new(16));
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        // Subscribe to events
        let mut event_rx = event_bus.subscribe();

        // Spawn telemetry loop
        let loop_handle = tokio::spawn(run_telemetry_loop(
            telemetry,
            state.clone(),
            event_bus.clone(),
            shutdown_rx,
        ));

        // Wait for a few GPU status events
        let mut event_count = 0;
        for _ in 0..3 {
            tokio::select! {
                event = event_rx.recv() => {
                    match event.unwrap() {
                        PrismEvent::GpuStatus { device_id, utilization, .. } => {
                            assert_eq!(device_id, 0);
                            assert!(utilization >= 0.0 && utilization <= 1.0);
                            event_count += 1;
                        }
                        _ => {}
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(200)) => {
                    break;
                }
            }
        }

        // Should have received at least one event
        assert!(event_count > 0);

        // Shutdown gracefully
        shutdown_tx.send(()).unwrap();
        loop_handle.await.unwrap().unwrap();
    }
}
