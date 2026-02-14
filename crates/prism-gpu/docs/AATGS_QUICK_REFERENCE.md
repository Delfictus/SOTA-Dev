# AATGS Quick Reference Guide

**Adaptive Asynchronous Task Graph Scheduler**
**Location:** `/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src/aatgs.rs`

---

## Quick Start

```rust
use prism_gpu::aatgs::{AATGSStreamIntegration, TaskType};
use cudarc::driver::CudaContext;
use std::sync::Arc;

// 1. Initialize
let device = CudaContext::new(0)?;
let mut aatgs = AATGSStreamIntegration::new(Arc::new(device))?;

// 2. Build task graph
let t0 = aatgs.add_task(TaskType::ConfigUpload, &[]);
let t1 = aatgs.add_task(TaskType::WhcrKernel, &[t0]);
let t2 = aatgs.add_task(TaskType::TelemetryDownload, &[t1]);

// 3. Execute
let config = RuntimeConfig::production();
let telemetry = aatgs.execute_iteration(config)?;

// 4. Monitor
let stats = aatgs.performance_stats();
println!("Pending: {}, Running: {}, Completed: {}",
         stats.tasks_pending, stats.tasks_running, stats.tasks_completed);

// 5. Shutdown
aatgs.shutdown()?;
```

---

## Core Components

### 1. ConfigCircularBuffer<T>

**Lock-free circular buffer for config upload**

```rust
let buffer = ConfigCircularBuffer::<RuntimeConfig>::new(16);

// Producer (CPU)
if buffer.push(config) {
    println!("Config queued");
}

// Consumer (GPU)
if let Some(config) = buffer.pop() {
    // Process config
}

// Monitor
println!("Utilization: {:.0}%", buffer.utilization() * 100.0);
```

**Key Methods:**
- `new(capacity)` - Create buffer
- `push(item) -> bool` - Add item (returns false if full)
- `pop() -> Option<T>` - Remove item
- `is_empty()` / `is_full()` - Check status
- `utilization() -> f32` - Get fill percentage

---

### 2. TelemetryCollector

**Async telemetry with event streaming**

```rust
let (mut collector, rx) = TelemetryCollector::new(64);

// Record metrics
collector.record_phase_metrics(PhaseMetrics {
    phase_id: 0,
    temperature: 1.5,
    compaction_ratio: 0.8,
    reward: 0.6,
    conflicts: 5,
    duration_us: 1000,
});

collector.record_gpu_metrics(GpuMetrics {
    utilization: 0.85,
    memory_used_mb: 2048,
    memory_total_mb: 8192,
    kernel_duration_us: 5000,
    transfer_duration_us: 100,
});

// Async event stream
while let Ok(event) = rx.try_recv() {
    match event {
        TelemetryEvent::PhaseMetrics(m) => println!("Phase {}: reward={}", m.phase_id, m.reward),
        TelemetryEvent::GpuMetrics(m) => println!("GPU: {:.0}%", m.utilization * 100.0),
        TelemetryEvent::TaskCompleted(id, dur) => println!("Task {} done in {:?}", id.0, dur),
        TelemetryEvent::Error(e) => eprintln!("Error: {}", e),
    }
}
```

---

### 3. TaskGraph

**DAG for task dependencies**

```rust
let mut graph = TaskGraph::new();

// Add tasks
let upload = graph.add_task(TaskType::ConfigUpload, &[]);
let whcr = graph.add_task(TaskType::WhcrKernel, &[upload]);
let thermo = graph.add_task(TaskType::ThermodynamicAnneal, &[whcr]);

// Execute
while let Some(task_id) = graph.pop_ready_task() {
    graph.mark_started(task_id);

    // ... execute task ...

    graph.mark_complete(task_id);
}

// Verify order
if let Some(sorted) = graph.topological_sort() {
    println!("Execution order: {:?}", sorted);
} else {
    eprintln!("Cycle detected!");
}
```

**Task Types:**
- `ConfigUpload` (100µs)
- `WhcrKernel` (5ms)
- `ThermodynamicAnneal` (10ms)
- `QuantumOptimize` (15ms)
- `LbsPredict` (8ms)
- `TelemetryDownload` (200µs)
- `PhaseTransition` (1ms)
- `Custom(&str)` (5ms)

---

### 4. AdaptiveScheduler

**Performance-based task scheduling**

```rust
let (mut scheduler, rx) = AdaptiveScheduler::new(16, 64, 100);

// Add tasks
let t0 = scheduler.add_task(TaskType::ConfigUpload, &[]);
let t1 = scheduler.add_task(TaskType::WhcrKernel, &[t0]);

// Schedule next (highest priority)
if let Some(task) = scheduler.schedule_next() {
    println!("Scheduling {:?} with priority {:.2}", task.task_type, task.priority);

    let start = Instant::now();
    scheduler.mark_task_started(task.task_id);

    // ... execute task ...

    scheduler.mark_task_completed(task.task_id, start.elapsed());
}

// Adapt based on performance
scheduler.adapt_priorities();

// Estimate completion
println!("ETA: {:?}", scheduler.estimate_completion_time());
```

**Priority Algorithm:**
- Base: 1.0
- Config boost: ×2.0 when buffer < 30%
- Telemetry boost: ×1.5 when count > 50
- Long task penalty: ×0.7 when GPU > 80%

---

### 5. AATGSStreamIntegration

**Unified interface**

```rust
let mut aatgs = AATGSStreamIntegration::new(device)?;

// Simple iteration
let config = RuntimeConfig::production();
if let Some(telemetry) = aatgs.execute_iteration(config)? {
    println!("Conflicts: {}", telemetry.conflicts);
}

// Performance monitoring
let stats = aatgs.performance_stats();
println!("Config buffer: {:.0}%", stats.config_buffer_utilization * 100.0);
println!("Telemetry buffer: {:.0}%", stats.telemetry_buffer_utilization * 100.0);
println!("Tasks: {} pending, {} running, {} done",
         stats.tasks_pending, stats.tasks_running, stats.tasks_completed);
println!("ETA: {:?}", stats.estimated_completion);
```

---

## Common Patterns

### Pattern 1: Linear Pipeline

```rust
let t0 = aatgs.add_task(TaskType::ConfigUpload, &[]);
let t1 = aatgs.add_task(TaskType::WhcrKernel, &[t0]);
let t2 = aatgs.add_task(TaskType::ThermodynamicAnneal, &[t1]);
let t3 = aatgs.add_task(TaskType::TelemetryDownload, &[t2]);
```

### Pattern 2: Diamond Dependency

```rust
let t0 = aatgs.add_task(TaskType::ConfigUpload, &[]);
let t1a = aatgs.add_task(TaskType::WhcrKernel, &[t0]);
let t1b = aatgs.add_task(TaskType::ThermodynamicAnneal, &[t0]);
let t2 = aatgs.add_task(TaskType::TelemetryDownload, &[t1a, t1b]);
```

### Pattern 3: Batch Processing

```rust
for _ in 0..10 {
    let t0 = aatgs.add_task(TaskType::ConfigUpload, &[]);
    let t1 = aatgs.add_task(TaskType::WhcrKernel, &[t0]);
    let t2 = aatgs.add_task(TaskType::TelemetryDownload, &[t1]);
}

// Execute all
while aatgs.performance_stats().tasks_pending > 0 {
    aatgs.execute_iteration(config)?;
}
```

### Pattern 4: Error Handling

```rust
match aatgs.execute_iteration(config) {
    Ok(Some(telemetry)) => {
        if telemetry.conflicts > 0 {
            println!("Warning: {} conflicts", telemetry.conflicts);
        }
    }
    Ok(None) => {
        println!("No telemetry available yet (pipeline filling)");
    }
    Err(e) => {
        eprintln!("AATGS error: {}", e);
        // Fallback logic
    }
}
```

---

## Performance Tips

### 1. Buffer Sizing

```rust
// Small graphs: 16 config slots, 64 telemetry slots (default)
let (scheduler, _) = AdaptiveScheduler::new(16, 64, 100);

// Large graphs: 64 config slots, 256 telemetry slots
let (scheduler, _) = AdaptiveScheduler::new(64, 256, 100);
```

### 2. Snapshot Tuning

```rust
// Frequent snapshots for fast-changing workloads
let (mut scheduler, _) = AdaptiveScheduler::new(16, 64, 100);
scheduler.snapshot_interval = Duration::from_millis(50);

// Infrequent snapshots for stable workloads
scheduler.snapshot_interval = Duration::from_millis(200);
```

### 3. History Management

```rust
// Short history for memory-constrained systems
let (scheduler, _) = AdaptiveScheduler::new(16, 64, 50);

// Long history for trend analysis
let (scheduler, _) = AdaptiveScheduler::new(16, 64, 500);
```

### 4. Batch Flushing

```rust
// Manual control for batching
let mut aatgs = AATGSScheduler::new(device)?;

for config in configs {
    aatgs.queue_config(config)?;  // Queue without flushing
}

aatgs.flush_configs()?;  // Flush all at once
```

---

## Integration Examples

### With Phase Controllers

```rust
impl PhaseController {
    fn execute_with_aatgs(&mut self, aatgs: &mut AATGSStreamIntegration) -> Result<()> {
        // Add phase-specific tasks
        let config_task = aatgs.add_task(TaskType::ConfigUpload, &[]);
        let whcr_task = aatgs.add_task(TaskType::WhcrKernel, &[config_task]);
        let telemetry_task = aatgs.add_task(TaskType::TelemetryDownload, &[whcr_task]);

        // Execute
        let config = self.build_config();
        let telemetry = aatgs.execute_iteration(config)?;

        // Update state
        if let Some(t) = telemetry {
            self.update_from_telemetry(t);
        }

        Ok(())
    }
}
```

### With FluxNet RL

```rust
impl FluxNetIntegration for AATGSStreamIntegration {
    fn record_reward(&mut self, phase_id: usize, reward: f32) {
        let metrics = PhaseMetrics {
            phase_id,
            reward,
            temperature: 0.0,
            compaction_ratio: 0.0,
            conflicts: 0,
            duration_us: 0,
        };

        // Access internal telemetry collector
        self.adaptive_scheduler_mut()
            .telemetry
            .record_phase_metrics(metrics);
    }
}
```

### With Multi-GPU

```rust
struct MultiGpuAATGS {
    schedulers: Vec<AATGSStreamIntegration>,
}

impl MultiGpuAATGS {
    fn new(num_devices: usize) -> Result<Self> {
        let schedulers = (0..num_devices)
            .map(|i| {
                let device = CudaContext::new(i)?;
                AATGSStreamIntegration::new(Arc::new(device))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { schedulers })
    }

    fn execute_all(&mut self, config: RuntimeConfig) -> Result<Vec<Option<KernelTelemetry>>> {
        self.schedulers
            .iter_mut()
            .map(|s| s.execute_iteration(config))
            .collect()
    }
}
```

---

## Debugging

### Enable Logging

```rust
env_logger::init();

// Logs from AATGS:
// [DEBUG] Flushing 5 configs to GPU
// [DEBUG] Polling 3 telemetry entries from GPU
// [DEBUG] Scheduling task WhcrKernel (priority: 1.50)
// [INFO] Signaling GPU shutdown
```

### Inspect Task Graph

```rust
let graph = aatgs.adaptive_scheduler().task_graph();

for (id, task_type, status) in graph.tasks() {
    println!("Task {}: {:?} - {:?}", id, task_type, status);
}

if let Some(sorted) = graph.topological_sort() {
    println!("Execution order: {:?}", sorted);
}
```

### Monitor Buffer Health

```rust
loop {
    let stats = aatgs.performance_stats();

    if stats.config_buffer_utilization > 0.9 {
        eprintln!("WARNING: Config buffer nearly full!");
    }

    if stats.telemetry_buffer_utilization > 0.9 {
        eprintln!("WARNING: Telemetry buffer nearly full!");
    }

    std::thread::sleep(Duration::from_millis(100));
}
```

---

## API Reference

### ConfigCircularBuffer<T>
- `new(capacity: usize) -> Self`
- `push(&self, item: T) -> bool`
- `pop(&self) -> Option<T>`
- `is_empty(&self) -> bool`
- `is_full(&self) -> bool`
- `len(&self) -> usize`
- `capacity(&self) -> usize`
- `utilization(&self) -> f32`

### TelemetryCollector
- `new(buffer_size: usize) -> (Self, Receiver<TelemetryEvent>)`
- `record_phase_metrics(&mut self, metrics: PhaseMetrics)`
- `record_gpu_metrics(&mut self, metrics: GpuMetrics)`
- `record_task_completion(&self, task_id: TaskId, duration: Duration)`
- `record_error(&self, error: String)`
- `flush(&mut self) -> Vec<TelemetryEvent>`
- `buffer_stats(&self) -> (usize, usize)`

### TaskGraph
- `new() -> Self`
- `add_task(&mut self, task_type: TaskType, dependencies: &[usize]) -> usize`
- `mark_started(&mut self, task_id: usize)`
- `mark_complete(&mut self, task_id: usize)`
- `mark_failed(&mut self, task_id: usize)`
- `get_ready_tasks(&self) -> Vec<usize>`
- `pop_ready_task(&mut self) -> Option<usize>`
- `topological_sort(&self) -> Option<Vec<usize>>`
- `task_status(&self, task_id: usize) -> Option<TaskStatus>`
- `task_duration(&self, task_id: usize) -> Option<Duration>`
- `tasks(&self) -> Vec<(usize, TaskType, TaskStatus)>`

### AdaptiveScheduler
- `new(config_buffer_size, telemetry_buffer_size, max_history) -> (Self, Receiver<TelemetryEvent>)`
- `add_task(&mut self, task_type: TaskType, dependencies: &[usize]) -> usize`
- `schedule_next(&mut self) -> Option<ScheduledTask>`
- `adapt_priorities(&mut self)`
- `estimate_completion_time(&self) -> Duration`
- `mark_task_started(&mut self, task_id: usize)`
- `mark_task_completed(&mut self, task_id: usize, duration: Duration)`
- `mark_task_failed(&mut self, task_id: usize, error: String)`
- `queue_config(&self, config: RuntimeConfig) -> bool`

### AATGSStreamIntegration
- `new(device: Arc<CudaContext>) -> Result<Self>`
- `execute_iteration(&mut self, config: RuntimeConfig) -> Result<Option<KernelTelemetry>>`
- `add_task(&mut self, task_type: TaskType, dependencies: &[usize]) -> usize`
- `performance_stats(&self) -> PerformanceStats`
- `estimate_completion_time(&self) -> Duration`
- `shutdown(self) -> Result<()>`

---

**Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.**
