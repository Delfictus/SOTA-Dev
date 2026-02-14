# Reactive Controller - Quick Reference

## Import

```rust
use prism::ui::{ReactiveController, ReactiveControllerBuilder};
use prism::runtime::{PrismRuntime, RuntimeConfig};
use prism::runtime::events::{PrismEvent, OptimizationConfig, ParameterValue};
```

## Setup

```rust
// Create runtime
let mut runtime = PrismRuntime::new(RuntimeConfig::default())?;
runtime.start().await?;

// Create controller
let event_rx = runtime.subscribe();
let (cmd_tx, _) = tokio::sync::mpsc::channel(256);
let mut controller = ReactiveController::new(
    event_rx,
    runtime.state.clone(),
    cmd_tx
);

// Create app
let mut app = App::new(None, "coloring".into(), 0)?;
```

## Main Loop

```rust
loop {
    // 1. Process runtime events → update app
    controller.poll_events(&mut app)?;

    // 2. Render UI
    terminal.draw(|f| app.render(f))?;

    // 3. Handle user input
    if crossterm::event::poll(Duration::from_millis(50))? {
        handle_input(&mut app, &controller).await?;
    }

    // 4. Check quit
    if app.should_quit { break; }
}
```

## Commands (User → Runtime)

```rust
// Load files
controller.load_graph("/data/graph.col".into()).await?;
controller.load_protein("/data/protein.pdb".into()).await?;

// Control optimization
let config = OptimizationConfig::default();
controller.start_optimization(config).await?;
controller.pause().await?;
controller.resume().await?;
controller.stop().await?;

// Set parameters
controller.set_parameter(
    "temperature".into(),
    ParameterValue::Float(1.5)
).await?;

// Shutdown
controller.shutdown().await?;
```

## State Access (Direct Queries)

```rust
// Get convergence history for plotting
let history = controller.get_convergence_history();
// Returns: Vec<(iteration: u64, colors: usize, conflicts: usize)>

// Get GPU utilization
let gpu_util = controller.get_gpu_utilization_history();
// Returns: Vec<(timestamp_ms: u64, utilization: f64)>

// Get temperature history
let temps = controller.get_temperature_history();
// Returns: Vec<(timestamp_ms: u64, replica_id: usize, temperature: f64)>

// Get statistics
let stats = controller.stats();
println!("Events/sec: {:.1}", controller.event_rate());
```

## Events (Runtime → UI) - Automatic

These are handled automatically by `poll_events()`:

| Event | App Update |
|-------|------------|
| `GraphLoaded` | `app.optimization.max_iterations` |
| `PhaseStarted` | `app.phases[idx].status = Running` |
| `PhaseProgress` | `app.optimization.{colors, conflicts, iteration, temperature}` |
| `PhaseCompleted` | `app.phases[idx].status = Completed` |
| `NewBestSolution` | `app.optimization.best_{colors,conflicts}` + dialogue |
| `GpuStatus` | `app.gpu.*` |
| `KernelLaunched` | `app.gpu.active_kernels.push()` |
| `ReplicaUpdate` | `app.optimization.replicas[idx].*` |
| `QuantumState` | `app.optimization.quantum_*` |
| `Error` | `app.dialogue.add_system_message()` |
| `Shutdown` | `app.should_quit = true` |

## Configuration

```rust
let controller = ReactiveControllerBuilder::new()
    .max_events_per_poll(100)      // Process up to 100 events/frame
    .poll_timeout_us(50)           // 50μs poll timeout
    .command_queue_capacity(512)   // 512 command buffer
    .build(event_rx, state, cmd_tx);
```

## Error Handling

```rust
// All operations return Result
match controller.poll_events(&mut app) {
    Ok(()) => {},
    Err(e) => eprintln!("Event error: {:?}", e),
}

// Commands can fail
if let Err(e) = controller.load_graph(path).await {
    app.dialogue.add_system_message(&format!("Load failed: {}", e));
}
```

## Performance Tuning

```rust
// High-throughput scenario (many events)
ReactiveControllerBuilder::new()
    .max_events_per_poll(200)  // Process more events per frame
    .build(...)

// Low-latency scenario (responsive UI)
ReactiveControllerBuilder::new()
    .max_events_per_poll(20)   // Fewer events, more render time
    .build(...)
```

## Complete Example

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Setup
    let mut runtime = PrismRuntime::new(RuntimeConfig::default())?;
    runtime.start().await?;

    let event_rx = runtime.subscribe();
    let (cmd_tx, _) = mpsc::channel(256);
    let mut controller = ReactiveController::new(event_rx, runtime.state.clone(), cmd_tx);
    let mut app = App::new(None, "coloring".into(), 0)?;
    let mut terminal = setup_terminal()?;

    // Main loop
    loop {
        controller.poll_events(&mut app)?;
        terminal.draw(|f| app.render(f))?;

        if crossterm::event::poll(Duration::from_millis(50))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                match key.code {
                    KeyCode::Char('r') => {
                        controller.start_optimization(OptimizationConfig::default()).await?;
                    }
                    KeyCode::Char('q') => break,
                    _ => {}
                }
            }
        }

        if app.should_quit { break; }
    }

    // Cleanup
    cleanup_terminal(&mut terminal)?;
    runtime.shutdown().await?;
    Ok(())
}
```

## Testing

```rust
#[tokio::test]
async fn test_reactive() {
    let event_bus = EventBus::new(16);
    let state = Arc::new(StateStore::new(100));
    let (cmd_tx, _) = mpsc::channel(16);

    let event_rx = event_bus.subscribe();
    let mut controller = ReactiveController::new(event_rx, state, cmd_tx);
    let mut app = App::new(None, "coloring".into(), 0)?;

    // Publish event
    event_bus.publish(PrismEvent::GraphLoaded {
        vertices: 500,
        edges: 12500,
        density: 0.1,
        estimated_chromatic: 48,
    }).await?;

    // Process event
    controller.poll_events(&mut app)?;

    // Verify
    assert_eq!(app.optimization.max_iterations, 48000);
}
```

## Key Files

- **Implementation**: `/mnt/c/Users/Predator/Desktop/PRISM/crates/prism/src/ui/reactive.rs` (648 LOC)
- **Integration Guide**: `/mnt/c/Users/Predator/Desktop/PRISM/crates/prism/src/ui/REACTIVE_INTEGRATION.md`
- **Summary**: `/mnt/c/Users/Predator/Desktop/PRISM/REACTIVE_CONTROLLER_SUMMARY.md`

## Common Patterns

### Pattern 1: Load and Run
```rust
controller.load_graph("/data/DSJC500.5.col".into()).await?;
tokio::time::sleep(Duration::from_millis(100)).await; // Wait for load
controller.start_optimization(OptimizationConfig::default()).await?;
```

### Pattern 2: Monitor Progress
```rust
loop {
    controller.poll_events(&mut app)?;

    if app.optimization.conflicts == 0 {
        println!("Valid coloring found: {} colors", app.optimization.colors);
        break;
    }

    terminal.draw(|f| render_progress(f, &app))?;
}
```

### Pattern 3: Adaptive Parameters
```rust
if app.optimization.iteration > 1000 && app.optimization.conflicts > 100 {
    controller.set_parameter(
        "temperature".into(),
        ParameterValue::Float(2.0)  // Increase temperature
    ).await?;
}
```

---

**See Also**: [Full Integration Guide](./REACTIVE_INTEGRATION.md) | [Runtime Docs](../runtime/README.md)
