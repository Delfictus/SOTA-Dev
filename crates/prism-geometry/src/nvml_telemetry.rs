//! NVML telemetry integration for GPU metrics during geometry stress analysis.
//!
//! Collects GPU utilization, memory usage, temperature, and power consumption
//! using NVIDIA Management Library (NVML).

use log::{debug, warn};
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// GPU metrics collected via NVML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage (0-100)
    pub gpu_utilization: u32,

    /// Memory utilization percentage (0-100)
    pub memory_utilization: u32,

    /// GPU temperature (Celsius)
    pub temperature: Option<u32>,

    /// Power consumption (milliwatts)
    pub power_usage: Option<u32>,

    /// Total memory (bytes)
    pub total_memory: u64,

    /// Used memory (bytes)
    pub used_memory: u64,

    /// Free memory (bytes)
    pub free_memory: u64,

    /// Timestamp when metrics were collected
    pub timestamp_ms: f64,
}

/// NVML telemetry collector with throttling to avoid overhead
pub struct NvmlTelemetry {
    nvml: Option<Nvml>,
    device_index: u32,
    last_sample_time: Option<Instant>,
    sample_interval_ms: u64,
}

impl NvmlTelemetry {
    /// Initialize NVML telemetry collector
    ///
    /// # Arguments
    /// * `device_index` - CUDA device index (usually 0)
    /// * `sample_interval_ms` - Minimum interval between samples (default: 100ms)
    ///
    /// # Returns
    /// NvmlTelemetry instance (with graceful fallback if NVML unavailable)
    pub fn new(device_index: u32, sample_interval_ms: u64) -> Self {
        let nvml = match Nvml::init() {
            Ok(nvml) => {
                debug!("NVML initialized successfully");
                Some(nvml)
            }
            Err(e) => {
                warn!("Failed to initialize NVML: {}. GPU metrics disabled.", e);
                None
            }
        };

        Self {
            nvml,
            device_index,
            last_sample_time: None,
            sample_interval_ms,
        }
    }

    /// Sample GPU metrics (with throttling)
    ///
    /// Returns None if:
    /// - NVML is unavailable
    /// - Sample interval hasn't elapsed
    /// - GPU query fails
    pub fn sample(&mut self) -> Option<GpuMetrics> {
        // Check if enough time has elapsed since last sample
        if let Some(last_time) = self.last_sample_time {
            if last_time.elapsed().as_millis() < self.sample_interval_ms as u128 {
                return None; // Throttle
            }
        }

        let nvml = self.nvml.as_ref()?;

        let device = match nvml.device_by_index(self.device_index) {
            Ok(dev) => dev,
            Err(e) => {
                warn!("Failed to get NVML device {}: {}", self.device_index, e);
                return None;
            }
        };

        // Query GPU utilization
        let utilization = match device.utilization_rates() {
            Ok(rates) => (rates.gpu, rates.memory),
            Err(e) => {
                debug!("Failed to query GPU utilization: {}", e);
                (0, 0)
            }
        };

        // Query memory info
        let (total_memory, used_memory, free_memory) = match device.memory_info() {
            Ok(info) => (info.total, info.used, info.free),
            Err(e) => {
                debug!("Failed to query GPU memory: {}", e);
                (0, 0, 0)
            }
        };

        // Query temperature (optional)
        let temperature = device
            .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
            .ok();

        // Query power usage (optional)
        let power_usage = device.power_usage().ok();

        self.last_sample_time = Some(Instant::now());

        Some(GpuMetrics {
            gpu_utilization: utilization.0,
            memory_utilization: utilization.1,
            temperature,
            power_usage,
            total_memory,
            used_memory,
            free_memory,
            timestamp_ms: Instant::now().elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Force sample immediately (ignore throttling)
    pub fn sample_now(&mut self) -> Option<GpuMetrics> {
        self.last_sample_time = None;
        self.sample()
    }

    /// Check if NVML is available
    pub fn is_available(&self) -> bool {
        self.nvml.is_some()
    }
}

impl Default for NvmlTelemetry {
    fn default() -> Self {
        Self::new(0, 100) // Device 0, 100ms interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvml_init() {
        let telemetry = NvmlTelemetry::new(0, 100);

        // Should not panic even if NVML unavailable
        if telemetry.is_available() {
            println!("NVML available: collecting metrics");
        } else {
            println!("NVML unavailable: metrics disabled (expected in CI)");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sample_metrics() {
        let mut telemetry = NvmlTelemetry::new(0, 0); // No throttling

        if let Some(metrics) = telemetry.sample() {
            println!("GPU Metrics: {:#?}", metrics);

            // Validate ranges
            assert!(metrics.gpu_utilization <= 100);
            assert!(metrics.memory_utilization <= 100);

            if let Some(temp) = metrics.temperature {
                assert!(temp > 0 && temp < 150, "Temperature out of range");
            }

            assert!(metrics.total_memory > 0);
        } else {
            panic!("NVML not available on GPU machine");
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_throttling() {
        use std::thread;
        use std::time::Duration;

        let mut telemetry = NvmlTelemetry::new(0, 200); // 200ms interval

        // First sample should succeed
        let first = telemetry.sample();
        assert!(first.is_some());

        // Immediate second sample should be throttled
        let second = telemetry.sample();
        assert!(second.is_none(), "Sample was not throttled");

        // After sleep, should succeed
        thread::sleep(Duration::from_millis(250));
        let third = telemetry.sample();
        assert!(third.is_some());
    }
}
