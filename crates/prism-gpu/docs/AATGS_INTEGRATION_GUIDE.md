# AATGS Integration Guide

## Overview

This guide shows how to wire the **AATGS (Adaptive Asynchronous Task Graph Scheduler)** into existing GPU modules for async execution.

## Architecture

### Traditional Sync Execution

```
CPU Thread                   GPU (CUDA)
──────────                   ──────────
upload_config() ─────────▶
                             kernel_launch()
                             kernel_execute()
                             kernel_finish()
download_results() ◀─────────
      │
      └─ BLOCKED waiting for GPU
```

**Problem**: CPU is idle during GPU execution

### AATGS Async Execution

```
CPU Thread                   GPU (CUDA)
──────────                   ──────────
queue_config(0) ─────────▶  [Buffer slot 0]
queue_config(1) ─────────▶  [Buffer slot 1]   kernel_execute(0)
queue_config(2) ─────────▶  [Buffer slot 2]   kernel_execute(1)
poll_telemetry() ◀───────── [Telemetry 0]     kernel_execute(2)
      │                                        kernel_execute(3)
      └─ NON-BLOCKING, CPU can continue
```

**Benefit**: CPU-GPU overlap, 1.5-3x throughput improvement

---

## Integration Steps

### Step 1: Wrap GPU Context

Replace raw `CudaDevice` with `GpuExecutionContext`:

**Before (Sync)**:
```rust
use cudarc::driver::CudaDevice;
use std::sync::Arc;

pub struct WhcrGpu {
    device: Arc<CudaDevice>,
    // ... other fields
}

impl WhcrGpu {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Initialize kernels
        Ok(Self { device })
    }
}
```

**After (Async-capable)**:
```rust
use prism_gpu::aatgs_integration::GpuExecutionContext;
use std::sync::Arc;

pub struct WhcrGpu {
    gpu_ctx: GpuExecutionContext,
    // ... other fields
}

impl WhcrGpu {
    pub fn new(device: Arc<CudaDevice>, enable_async: bool) -> Result<Self> {
        let gpu_ctx = GpuExecutionContext::new(device, enable_async)?;
        Ok(Self { gpu_ctx })
    }
}
```

### Step 2: Replace Kernel Launch with Execute

**Before (Sync)**:
```rust
pub fn repair_iteration(&mut self, config: RuntimeConfig) -> Result<KernelTelemetry> {
    // Upload config
    let d_config = self.device.htod_copy(vec![config])?;

    // Launch kernel
    let cfg = LaunchConfig::for_num_elems(self.num_vertices as u32);
    unsafe {
        self.repair_kernel.launch(cfg, (&d_config, &self.d_coloring, ...))?;
    }

    // Download telemetry
    let telemetry = self.device.dtoh_sync_copy(&self.d_telemetry)?;
    Ok(telemetry[0])
}
```

**After (Async-capable)**:
```rust
pub fn repair_iteration(&mut self, config: RuntimeConfig) -> Result<Option<KernelTelemetry>> {
    // Queue config and poll telemetry (non-blocking)
    self.gpu_ctx.execute(config)
}
```

### Step 3: Handle Pipeline Latency

AATGS has 1-2 iteration latency due to pipelining. Handle `None` results:

```rust
pub fn repair(&mut self, num_iterations: usize) -> Result<RepairResult> {
    let mut latest_telemetry = None;

    for iter in 0..num_iterations {
        let mut config = RuntimeConfig::production();
        config.iteration = iter as i32;

        // Execute (may return None for first few iterations)
        if let Some(telemetry) = self.gpu_ctx.execute(config)? {
            latest_telemetry = Some(telemetry);

            // Check for early termination
            if telemetry.conflicts == 0 {
                break;
            }
        }
    }

    // Flush pipeline to get final results
    self.gpu_ctx.flush()?;

    // Poll remaining telemetry
    while let Some(telemetry) = /* poll logic */ {
        latest_telemetry = Some(telemetry);
    }

    Ok(RepairResult {
        success: latest_telemetry.map(|t| t.conflicts == 0).unwrap_or(false),
        // ... other fields
    })
}
```

### Step 4: Batch Processing

For maximum throughput, use batch execution:

```rust
pub fn repair_batch(&mut self, configs: &[RuntimeConfig]) -> Result<Vec<KernelTelemetry>> {
    self.gpu_ctx.execute_batch(configs)
}
```

---

## Complete Example: WHCR Integration

```rust
use anyhow::Result;
use cudarc::driver::CudaDevice;
use prism_core::{RuntimeConfig, KernelTelemetry};
use prism_gpu::aatgs_integration::GpuExecutionContext;
use std::sync::Arc;

pub struct WhcrGpu {
    gpu_ctx: GpuExecutionContext,
    // ... kernel functions, buffers, etc.
}

impl WhcrGpu {
    pub fn new(device: Arc<CudaDevice>, enable_async: bool) -> Result<Self> {
        let gpu_ctx = GpuExecutionContext::new(device, enable_async)?;

        // Load PTX kernels, allocate buffers, etc.

        Ok(Self { gpu_ctx })
    }

    /// Repair with async execution
    pub fn repair_async(
        &mut self,
        coloring: &mut [usize],
        num_colors: usize,
        max_iterations: usize,
    ) -> Result<RepairResult> {
        let mut latest_telemetry = None;
        let mut total_iterations = 0;

        // Main repair loop
        for iter in 0..max_iterations {
            let mut config = RuntimeConfig::production();
            config.iteration = iter as i32;
            config.phase_id = 3; // WHCR phase

            // Queue work (non-blocking)
            if let Some(telemetry) = self.gpu_ctx.execute(config)? {
                latest_telemetry = Some(telemetry);
                total_iterations = iter + 1;

                // Early termination
                if telemetry.conflicts == 0 {
                    log::info!("WHCR converged at iteration {}", iter);
                    break;
                }
            }
        }

        // Flush pipeline to ensure all work completes
        self.gpu_ctx.flush()?;

        // Download final coloring from GPU
        // (This part depends on your specific buffer layout)

        let final_telemetry = latest_telemetry.unwrap_or_default();

        Ok(RepairResult {
            success: final_telemetry.conflicts == 0,
            final_colors: num_colors,
            final_conflicts: final_telemetry.conflicts as usize,
            iterations: total_iterations,
        })
    }
}

pub struct RepairResult {
    pub success: bool,
    pub final_colors: usize,
    pub final_conflicts: usize,
    pub iterations: usize,
}
```

---

## Performance Tuning

### Buffer Utilization

Monitor buffer stats to ensure efficient utilization:

```rust
let stats = gpu_ctx.stats();
println!("Config buffer peak: {:.1}%", stats.peak_config_util * 100.0);
println!("Telemetry buffer peak: {:.1}%", stats.peak_telemetry_util * 100.0);

// Ideal: 40-80% utilization
// Too low (<20%): Underutilized, consider batching
// Too high (>90%): Risk of overflows, consider increasing buffer size
```

### Buffer Overflow Handling

If overflows occur frequently:

```rust
if stats.buffer_overflows > 0 {
    log::warn!("Buffer overflows detected: {}", stats.buffer_overflows);
    // Option 1: Reduce queue rate
    // Option 2: Increase buffer sizes in aatgs.rs (CONFIG_BUFFER_SIZE, TELEMETRY_BUFFER_SIZE)
}
```

### GPU Idle Detection

Check if GPU is waiting for work:

```rust
if gpu_ctx.is_gpu_idle()? {
    log::warn!("GPU idle - CPU not feeding work fast enough");
    // Consider pre-computing more configs or batching
}
```

---

## Migration Checklist

- [ ] Replace `Arc<CudaDevice>` with `GpuExecutionContext` in struct
- [ ] Add `enable_async: bool` parameter to constructor
- [ ] Replace manual kernel launches with `ctx.execute(config)`
- [ ] Handle `Option<KernelTelemetry>` for pipeline latency
- [ ] Add `flush()` before final result retrieval
- [ ] Update tests to use both sync and async modes
- [ ] Benchmark and compare throughput
- [ ] Monitor buffer utilization in production

---

## Troubleshooting

### Issue: Pipeline never returns telemetry

**Cause**: GPU kernel not writing to telemetry buffer

**Fix**: Ensure kernel writes telemetry and updates `telemetry_write_ptr`:
```cuda
__global__ void my_kernel(..., AATGSBuffers* buffers) {
    // ... kernel logic ...

    // Write telemetry
    int slot = atomicAdd(&buffers->telemetry_write_ptr, 1) % TELEMETRY_BUFFER_SIZE;
    buffers->telemetry_buffer[slot] = telemetry;
}
```

### Issue: Lower performance than sync mode

**Cause**: Overhead dominates for fast kernels (<100μs)

**Fix**: AATGS is designed for iterative algorithms with kernels >1ms. For fast kernels, use sync mode.

### Issue: Buffer overflow errors

**Cause**: CPU queuing faster than GPU processing

**Fix**: Increase buffer sizes in `aatgs.rs`:
```rust
const CONFIG_BUFFER_SIZE: usize = 32; // Default: 16
const TELEMETRY_BUFFER_SIZE: usize = 128; // Default: 64
```

---

## Next Steps

1. **Wire to LBS**: Integrate with ligand binding site prediction
2. **Wire to Thermodynamic**: Async parallel tempering
3. **Wire to Active Inference**: Async belief updates
4. **Multi-GPU**: Extend AATGS for multi-device scheduling

---

## References

- `crates/prism-gpu/src/aatgs.rs` - Scheduler implementation
- `crates/prism-gpu/src/aatgs_integration.rs` - Integration layer
- `crates/prism-gpu/examples/aatgs_whcr_integration.rs` - Example usage
- PRISM Phase 0.5 Plan - Technical debt resolution
