# Stream Manager Integration Guide

## Overview

The `ManagedGpuContext` provides optional triple-buffered execution for GPU kernels, enabling ~2-3x throughput by overlapping:
1. Config upload (H2D transfer)
2. Kernel execution (GPU compute)
3. Telemetry download (D2H transfer)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ManagedGpuContext                         │
├─────────────────────────────────────────────────────────────┤
│  device: Arc<CudaDevice>                                    │
│  stream_pool: Option<StreamPool>          ← Stream manager  │
│  pipeline_coordinator: Option<...>        ← Triple-buffer   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Create Managed Context

```rust
use prism_gpu::ManagedGpuContext;
use cudarc::driver::CudaDevice;

// Synchronous mode (default)
let device = CudaDevice::new(0)?;
let ctx = ManagedGpuContext::new(device, false)?;

// Asynchronous mode (triple-buffered)
let device = CudaDevice::new(0)?;
let mut ctx = ManagedGpuContext::new(device, true)?;
```

### 2. Use Triple-Buffered Execution

```rust
use prism_core::RuntimeConfig;

// Single iteration
let config = RuntimeConfig::default();
let telemetry = ctx.triple_buffered_step(config)?;

// Multiple iterations (overlaps stages)
for iter in 0..1000 {
    let config = RuntimeConfig::default();
    let telemetry = ctx.triple_buffered_step(config)?;
    // Process telemetry...
}
```

### 3. Access Underlying Device

```rust
// Get device for kernel launches
let device = ctx.device();

// Get stream for async operations
if let Some(stream_idx) = ctx.get_stream(StreamPurpose::KernelExecution) {
    // Launch kernel on specific stream
}
```

## Integration Patterns

### Pattern 1: Retrofit Existing GPU Modules (Minimal Changes)

**Before:**
```rust
pub struct WhcrGpu {
    device: Arc<CudaDevice>,
    // ... kernels and buffers
}

impl WhcrGpu {
    pub fn repair(&mut self) -> Result<RepairResult> {
        // Launch kernels synchronously
        unsafe { self.kernel.launch(cfg, params)? };
        self.device.synchronize()?;
        // ...
    }
}
```

**After (with optional stream support):**
```rust
pub struct WhcrGpu {
    device: Arc<CudaDevice>,
    managed_ctx: Option<ManagedGpuContext>, // Optional managed context
    // ... kernels and buffers
}

impl WhcrGpu {
    pub fn new(device: Arc<CudaDevice>, enable_streams: bool) -> Result<Self> {
        let managed_ctx = if enable_streams {
            Some(ManagedGpuContext::new(device.clone(), true)?)
        } else {
            None
        };

        // ... rest of initialization
        Ok(Self { device, managed_ctx, /* ... */ })
    }

    pub fn repair(&mut self) -> Result<RepairResult> {
        if let Some(ref mut ctx) = self.managed_ctx {
            // Triple-buffered async path
            let config = RuntimeConfig::default();
            let _telemetry = ctx.triple_buffered_step(config)?;
            // ... use async execution
        } else {
            // Synchronous fallback (existing behavior)
            unsafe { self.kernel.launch(cfg, params)? };
            self.device.synchronize()?;
        }
        // ...
    }
}
```

### Pattern 2: New GPU Module (Built-in Stream Support)

```rust
use prism_gpu::{ManagedGpuContext, StreamPurpose};

pub struct NewGpuModule {
    ctx: ManagedGpuContext,
    kernel: CudaFunction,
}

impl NewGpuModule {
    pub fn new(device: Arc<CudaDevice>, enable_async: bool) -> Result<Self> {
        let ctx = ManagedGpuContext::new(device.clone(), enable_async)?;

        // Load PTX and get kernel
        device.load_ptx(ptx_str.into(), "module_name", &["kernel_name"])?;
        let kernel = device.get_func("module_name", "kernel_name")
            .ok_or_else(|| anyhow::anyhow!("Kernel not found"))?;

        Ok(Self { ctx, kernel })
    }

    pub fn execute(&mut self) -> Result<()> {
        if self.ctx.has_stream_management() {
            // Async execution
            let stream_idx = self.ctx.get_stream(StreamPurpose::KernelExecution)
                .ok_or_else(|| anyhow::anyhow!("Stream not available"))?;

            // Launch async
            unsafe {
                self.kernel.clone().launch_async(
                    LaunchConfig::default(),
                    params,
                    stream_idx
                )?
            };
        } else {
            // Sync execution
            unsafe { self.kernel.clone().launch(LaunchConfig::default(), params)? };
            self.ctx.device().synchronize()?;
        }
        Ok(())
    }
}
```

### Pattern 3: High-Level Pipeline Coordinator

```rust
pub struct PrismPipeline {
    whcr_ctx: ManagedGpuContext,
    thermo_ctx: ManagedGpuContext,
    quantum_ctx: ManagedGpuContext,
}

impl PrismPipeline {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            whcr_ctx: ManagedGpuContext::new(device.clone(), true)?,
            thermo_ctx: ManagedGpuContext::new(device.clone(), true)?,
            quantum_ctx: ManagedGpuContext::new(device.clone(), true)?,
        })
    }

    pub fn run_phase_2(&mut self) -> Result<()> {
        // Run thermodynamic phase with triple-buffering
        let config = RuntimeConfig::default();
        let telemetry = self.thermo_ctx.triple_buffered_step(config)?;

        // Process telemetry and continue...
        Ok(())
    }
}
```

## Stream Purposes

The `StreamPurpose` enum organizes streams by function:

```rust
pub enum StreamPurpose {
    ConfigUpload,        // H2D config transfers
    KernelExecution,     // Main compute kernels
    TelemetryDownload,   // D2H telemetry transfers
    P2PTransfer,         // Multi-GPU P2P copies
    AuxCompute,          // Auxiliary kernels
}
```

## Performance Considerations

### When to Use Async Mode

**Use async mode when:**
- Kernel execution time ≈ data transfer time (balanced workload)
- Running many iterations sequentially
- Multiple phases can overlap
- Target throughput > latency

**Use sync mode when:**
- Single iteration or very few iterations
- Kernel dominates (transfer negligible)
- Debugging or profiling
- Simpler code preferred

### Expected Speedup

| Kernel/Transfer Ratio | Async Speedup |
|-----------------------|---------------|
| 1:1 (balanced)        | 2.5-3x        |
| 10:1 (compute-bound)  | 1.1-1.2x      |
| 1:10 (transfer-bound) | 1.5-2x        |

### Memory Overhead

Triple-buffering requires 3x config/telemetry buffers:
- Config buffer: ~1KB per iteration
- Telemetry buffer: ~10KB per iteration
- Total overhead: ~33KB (negligible)

## Integration Checklist

When retrofitting existing GPU modules:

- [ ] Add `ManagedGpuContext` as optional field
- [ ] Update constructor to accept `enable_streams: bool`
- [ ] Branch execution on `has_stream_management()`
- [ ] Keep synchronous path for backward compatibility
- [ ] Add tests for both sync and async modes
- [ ] Document async mode in module docs
- [ ] Update benchmarks to measure speedup

## Examples

### WHCR Integration

```rust
// In prism-gpu/src/whcr.rs
impl WhcrGpu {
    pub fn new_with_streams(
        device: Arc<CudaDevice>,
        ptx_path: &str,
        enable_streams: bool
    ) -> Result<Self> {
        let managed_ctx = if enable_streams {
            Some(ManagedGpuContext::new(device.clone(), true)?)
        } else {
            None
        };

        // ... existing initialization

        Ok(Self {
            device,
            managed_ctx,
            // ... existing fields
        })
    }
}
```

### Thermodynamic Integration

```rust
// In prism-gpu/src/thermodynamic.rs
impl ThermodynamicGpu {
    pub fn run_async(
        &mut self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
        initial_colors: &[usize],
        num_replicas: usize,
        iterations: usize,
    ) -> Result<Vec<usize>> {
        if let Some(ref mut ctx) = self.managed_ctx {
            // Triple-buffered execution
            for iter in 0..iterations {
                let config = RuntimeConfig::default();
                let telemetry = ctx.triple_buffered_step(config)?;
                // ... process telemetry
            }
        } else {
            // Synchronous fallback
            self.run(adjacency, num_vertices, initial_colors, num_replicas, iterations)
        }
    }
}
```

## Future Enhancements

1. **Stream priority**: High-priority streams for critical kernels
2. **Dynamic scheduling**: Auto-adjust stream allocation based on load
3. **Multi-GPU streams**: Per-device stream pools in multi-GPU setups
4. **Telemetry**: Track stream utilization and overlap efficiency
5. **Graph capture**: CUDA graph optimization for repeated patterns

## References

- `crates/prism-gpu/src/stream_manager.rs` - Core stream management
- `crates/prism-gpu/src/stream_integration.rs` - ManagedGpuContext
- `crates/prism-gpu/src/context.rs` - Base GPU context
- PRISM GPU Plan §4.3 - Stream Management Architecture
