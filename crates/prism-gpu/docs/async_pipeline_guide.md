# Async Pipeline Integration Guide

## Overview

The `stream_integration.rs` module provides a complete async pipeline framework for GPU operations with triple buffering, event-based synchronization, and overlapped execution.

## Components

### 1. TripleBuffer (GPU-Side)

GPU-side triple buffering for overlapping upload/compute/download:

```rust
use prism_gpu::GpuTripleBuffer;
use std::sync::Arc;

// Create triple buffer
let ctx = CudaContext::new(0)?;
let buffer = GpuTripleBuffer::<f32>::new(&Arc::new(ctx), 1024)?;

// Get buffers for different stages
let upload_buf = buffer.get_upload_buffer();    // Returns Some if available
let process_buf = buffer.get_process_buffer();  // Returns Some when upload done
let download_buf = buffer.get_download_buffer(); // Returns Some when processing done

// Advance the rotation
buffer.advance();
```

**State Machine:**
```
Free → Uploading → Processing → Downloading → Free
```

### 2. AsyncCoordinator

Event-based coordination of async GPU operations:

```rust
use prism_gpu::AsyncCoordinator;

let ctx = Arc::new(CudaContext::new(0)?);
let mut coordinator = AsyncCoordinator::new(ctx)?;

// Queue operations
let upload_id = coordinator.queue_upload(&host_data, &buffer)?;
let compute_id = coordinator.queue_compute("my_kernel", &[upload_id])?;
let download_id = coordinator.queue_download(&buffer, &mut host_dst)?;

// Non-blocking poll
let completed = coordinator.poll();

// Blocking wait
coordinator.wait(download_id)?;
```

**Event Dependencies:**
```
Upload Stream   ──[Event 0]──► Compute Stream   ──[Event 1]──► Download Stream
   (H2D)                          (Kernel)                         (D2H)
```

### 3. PipelineStageManager

Multi-stage pipeline execution with automatic overlapping:

```rust
use prism_gpu::{PipelineStageManager, GpuPipelineStage};
use cudarc::driver::LaunchConfig;

let coordinator = AsyncCoordinator::new(ctx)?;
let mut pipeline = PipelineStageManager::new(coordinator);

// Define stages
let preprocess = GpuPipelineStage {
    name: "Preprocess".to_string(),
    input_buffers: vec![],
    output_buffers: vec![0],
    config: LaunchConfig {
        grid_dim: (16, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    },
};

let compute = GpuPipelineStage {
    name: "MainCompute".to_string(),
    input_buffers: vec![0],
    output_buffers: vec![1],
    config: LaunchConfig {
        grid_dim: (32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 4096,
    },
};

// Add stages and set execution order
let idx0 = pipeline.add_stage(preprocess);
let idx1 = pipeline.add_stage(compute);
pipeline.set_order(vec![idx0, idx1]);

// Execute iterations
for _ in 0..100 {
    pipeline.execute_iteration()?;
}

// Get performance stats
let stats = pipeline.get_stats();
println!("Throughput: {:.2} iterations/sec", stats.throughput);
println!("Avg iteration: {:.2} ms", stats.avg_iteration_ms);
```

### 4. ManagedGpuContext

High-level interface integrating all components:

```rust
use prism_gpu::ManagedGpuContext;

// Create context with async enabled
let mut ctx = ManagedGpuContext::new(device, true)?;

// Create GPU triple buffer
let buffer = ctx.create_triple_buffer::<f32>(2048)?;

// Create pipeline manager
let mut pipeline = ctx.create_pipeline_manager()?;

// Access async coordinator
if let Some(coordinator) = ctx.async_coordinator_mut() {
    coordinator.queue_upload(&data, &buffer)?;
}

// Traditional triple-buffered step (uses AsyncPipelineCoordinator)
let telemetry = ctx.triple_buffered_step(config)?;
```

## Performance Characteristics

### Triple Buffering Speedup

| Operation | Time (ms) | Overlap Potential |
|-----------|-----------|-------------------|
| Upload    | 2.0       | 100%             |
| Compute   | 5.0       | 100%             |
| Download  | 2.0       | 100%             |

**Sequential:** 9.0 ms/iter → **111 iterations/sec**
**Triple-buffered:** 5.0 ms/iter → **200 iterations/sec** (1.8x speedup)

### Event Overhead

- Event creation: ~1 μs
- Event recording: ~0.5 μs
- Event query: ~0.2 μs
- Event wait: ~0.3 μs

**Total overhead per operation:** ~2 μs (negligible for kernels >100 μs)

## Usage Patterns

### Pattern 1: Simple Triple Buffering

```rust
let buffer = ctx.create_triple_buffer::<f32>(1024)?;
let mut coordinator = AsyncCoordinator::new(ctx.device().clone())?;

for iteration in 0..100 {
    // Upload next batch
    coordinator.queue_upload(&data[iteration], &buffer)?;

    // Process current batch (waits for upload)
    coordinator.queue_compute("kernel", &[0])?;

    // Download previous results (waits for compute)
    coordinator.queue_download(&buffer, &mut results[iteration])?;

    buffer.advance();
}
```

### Pattern 2: Multi-Stage Pipeline

```rust
let mut pipeline = ctx.create_pipeline_manager()?;

// Define 3-stage pipeline
pipeline.add_stage(stage_preprocess);
pipeline.add_stage(stage_compute);
pipeline.add_stage(stage_postprocess);
pipeline.set_order(vec![0, 1, 2]);

// Execute with automatic overlapping
for _ in 0..1000 {
    pipeline.execute_iteration()?;
}
```

### Pattern 3: Custom Event Dependencies

```rust
let mut coordinator = AsyncCoordinator::new(ctx)?;

// Fork: Two independent operations after upload
let upload_id = coordinator.queue_upload(&data, &buffer)?;
let compute_a = coordinator.queue_compute("kernel_a", &[upload_id])?;
let compute_b = coordinator.queue_compute("kernel_b", &[upload_id])?;

// Join: Download waits for both
let download = coordinator.queue_compute("merge", &[compute_a, compute_b])?;
coordinator.wait(download)?;
```

## Integration with Existing Code

### Replace Synchronous Execution

**Before:**
```rust
device.htod_copy_into(&host_data, &device_buf)?;
kernel.launch(config, (&device_buf,))?;
device.dtoh_sync_copy_into(&device_buf, &mut host_result)?;
device.synchronize()?;
```

**After:**
```rust
let buffer = ctx.create_triple_buffer(size)?;
let mut coord = ctx.async_coordinator_mut().unwrap();

let upload = coord.queue_upload(&host_data, &buffer)?;
let compute = coord.queue_compute("kernel", &[upload])?;
coord.queue_download(&buffer, &mut host_result)?;
// Non-blocking! Can do CPU work here
coord.wait(compute)?;
```

### Add to StreamManager

The `AsyncCoordinator` is automatically created when stream management is enabled:

```rust
let ctx = ManagedGpuContext::new(device, true)?; // enable_streams = true

// AsyncCoordinator is available
let coordinator = ctx.async_coordinator().unwrap();
let upload_stream = coordinator.upload_stream();
let compute_stream = coordinator.compute_stream();
```

## Testing

Run GPU-accelerated tests:

```bash
cargo test --features cuda --package prism-gpu -- --ignored test_triple_buffer
cargo test --features cuda --package prism-gpu -- --ignored test_async_coordinator
cargo test --features cuda --package prism-gpu -- --ignored test_pipeline_stage_manager
```

## API Reference

### TripleBuffer\<T>

**Methods:**
- `new(device, size)` - Create with GPU memory
- `get_upload_buffer()` - Get buffer for upload (Free → Uploading)
- `get_process_buffer()` - Get buffer for processing (Uploading → Processing)
- `get_download_buffer()` - Get buffer for download (Processing → Downloading)
- `advance()` - Rotate buffers (Downloading → Free, rotate indices)
- `size()` - Get buffer size
- `get_states()` - Debug: view all buffer states

### AsyncCoordinator

**Methods:**
- `new(context)` - Create with dedicated streams
- `queue_upload(data, buffer)` - Async H2D copy, returns op_id
- `queue_compute(name, deps)` - Record compute op with dependencies
- `queue_download(buffer, dst)` - Async D2H copy
- `poll()` - Non-blocking check for completed ops
- `wait(op_id)` - Blocking wait for specific operation
- `synchronize_all()` - Wait for all streams

**Stream Access:**
- `compute_stream()` - Get compute stream
- `upload_stream()` - Get upload stream
- `download_stream()` - Get download stream

### PipelineStageManager

**Methods:**
- `new(coordinator)` - Create manager
- `add_stage(stage)` - Add pipeline stage, returns index
- `set_order(indices)` - Define execution order
- `execute_iteration()` - Run one full pipeline iteration
- `get_stats()` - Get performance statistics
- `reset_stats()` - Reset counters
- `coordinator()` - Access underlying AsyncCoordinator

### PipelineStats

**Fields:**
- `iterations: usize` - Total iterations executed
- `avg_iteration_ms: f64` - Running average iteration time
- `throughput: f64` - Iterations per second
- `gpu_utilization: f64` - GPU utilization (0.0-1.0)

## Thread Safety

- `TripleBuffer`: Uses atomic state tracking, safe for concurrent access
- `AsyncCoordinator`: Not Send/Sync (CUDA stream handles), use via `&mut`
- `PipelineStageManager`: Not Send/Sync (contains AsyncCoordinator)
- `ManagedGpuContext`: Send via `Arc<CudaContext>`, coordinators via `&mut`

## Memory Management

- **GPU Buffers:** Automatically freed when `TripleBuffer` is dropped
- **Events:** Pre-allocated pool (10 events), reused via modulo indexing
- **Streams:** Managed by CUDA, cleaned up on context drop

## Error Handling

All operations return `Result<T>`:
- `queue_upload` - Buffer unavailable, copy failure
- `queue_compute` - Invalid dependencies
- `queue_download` - Buffer unavailable, copy failure
- `wait` - Invalid operation ID, synchronization failure

## Performance Tips

1. **Pre-allocate buffers** - Create `TripleBuffer` once, reuse across iterations
2. **Minimize dependencies** - Only specify necessary wait_for dependencies
3. **Use poll() for CPU work** - Check completion while doing CPU tasks
4. **Pipeline similar kernels** - Group kernels with similar execution times
5. **Monitor stats** - Use `PipelineStats` to identify bottlenecks

## Debugging

Enable CUDA API tracing:
```bash
export CUDA_LAUNCH_BLOCKING=1
cargo run --features cuda
```

Check buffer states:
```rust
let states = buffer.get_states();
println!("Buffer states: {:?}", states);
```

Monitor event completion:
```rust
loop {
    let completed = coordinator.poll();
    for op in completed {
        println!("Completed: {} ({})", op.op_id, op.op_type);
    }
}
```

## Copyright

```
Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
Los Angeles, CA 90013
Contact: IS@Delfictus.com
All Rights Reserved.
```
