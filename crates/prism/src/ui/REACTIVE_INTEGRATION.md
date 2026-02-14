# Reactive Controller Integration Guide

This guide shows how to integrate the `ReactiveController` with the PRISM TUI application.

## Overview

The `ReactiveController` provides a bidirectional bridge between the PRISM runtime and the TUI:

```
Runtime Actors → Events → ReactiveController → App State Updates
User Actions ← Commands ← ReactiveController ← App Input
```

## Basic Integration

### 1. Initialize the Runtime and Controller

```rust
use prism::runtime::{PrismRuntime, RuntimeConfig};
use prism::ui::{App, ReactiveController};

#[tokio::main]
async fn main() -> Result<()> {
    // Create runtime
    let mut runtime = PrismRuntime::new(RuntimeConfig::default())?;
    runtime.start().await?;

    // Subscribe to runtime events
    let event_rx = runtime.subscribe();
    let state = runtime.state.clone();
    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::channel(256);

    // Create reactive controller
    let mut controller = ReactiveController::new(event_rx, state, cmd_tx);

    // Create TUI app
    let mut app = App::new(None, "coloring".into(), 0)?;

    // Run event loop (see below)
    run_ui_loop(&mut app, &mut controller, &mut runtime).await?;

    Ok(())
}
```

### 2. Event Loop Integration

The reactive controller should be polled in your main render loop:

```rust
async fn run_ui_loop(
    app: &mut App,
    controller: &mut ReactiveController,
    runtime: &mut PrismRuntime,
) -> Result<()> {
    let mut terminal = setup_terminal()?;

    loop {
        // 1. Poll runtime events and update app state
        controller.poll_events(app)?;

        // 2. Render UI
        terminal.draw(|frame| app.render(frame))?;

        // 3. Handle user input
        if crossterm::event::poll(Duration::from_millis(50))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                handle_key_event(key, app, controller).await?;
            }
        }

        // 4. Check for quit
        if app.should_quit {
            break;
        }
    }

    cleanup_terminal(&mut terminal)?;
    runtime.shutdown().await?;
    Ok(())
}
```

### 3. Handling User Commands

Convert user actions into runtime commands:

```rust
async fn handle_key_event(
    key: KeyEvent,
    app: &mut App,
    controller: &ReactiveController,
) -> Result<()> {
    match (key.modifiers, key.code) {
        // Load graph file
        (KeyModifiers::NONE, KeyCode::Char('l')) => {
            let path = prompt_for_file()?;
            controller.load_graph(path).await?;
        }

        // Start optimization
        (KeyModifiers::NONE, KeyCode::Char('r')) => {
            let config = OptimizationConfig::default();
            controller.start_optimization(config).await?;
        }

        // Pause/Resume
        (KeyModifiers::NONE, KeyCode::Char('p')) => {
            controller.pause().await?;
        }
        (KeyModifiers::NONE, KeyCode::Char('c')) => {
            controller.resume().await?;
        }

        // Stop
        (KeyModifiers::NONE, KeyCode::Char('s')) => {
            controller.stop().await?;
        }

        // Set parameter
        (KeyModifiers::NONE, KeyCode::Char('t')) => {
            use prism::runtime::events::ParameterValue;
            let temp = prompt_for_temperature()?;
            controller.set_parameter(
                "temperature".into(),
                ParameterValue::Float(temp)
            ).await?;
        }

        _ => {}
    }

    Ok(())
}
```

## Advanced Usage

### Custom Configuration

```rust
use prism::ui::{ReactiveControllerBuilder, ReactiveConfig};

let controller = ReactiveControllerBuilder::new()
    .max_events_per_poll(100)        // Process up to 100 events per frame
    .poll_timeout_us(50)             // 50 microsecond poll timeout
    .command_queue_capacity(512)     // Larger command queue
    .build(event_rx, state, cmd_tx);
```

### Accessing State Store Directly

For high-frequency data that doesn't need to trigger UI updates:

```rust
// Get convergence history for plotting
let history = controller.get_convergence_history();
for (iteration, colors, conflicts) in history {
    plot_point(iteration, colors);
}

// Get GPU utilization for metrics
let gpu_util = controller.get_gpu_utilization_history();

// Get temperature history for thermodynamic visualization
let temps = controller.get_temperature_history();
```

### Statistics and Monitoring

```rust
// Get controller performance stats
let stats = controller.stats();
println!("Events processed: {}", stats.events_processed);
println!("Event rate: {:.1} events/sec", controller.event_rate());
```

## Event Flow

### Runtime → UI (Automatic)

These events are automatically handled by `poll_events()`:

| Event Type | Effect on App |
|------------|---------------|
| `GraphLoaded` | Updates `app.optimization.max_iterations` |
| `PhaseStarted` | Sets `app.phases[idx].status = Running` |
| `PhaseProgress` | Updates `app.optimization` and convergence history |
| `PhaseCompleted` | Sets `app.phases[idx].status = Completed` |
| `NewBestSolution` | Updates `app.optimization.best_colors/conflicts` |
| `GpuStatus` | Updates `app.gpu.*` fields |
| `ReplicaUpdate` | Updates `app.optimization.replicas` |
| `QuantumState` | Updates `app.optimization.quantum_*` |
| `Error` | Adds error message to `app.dialogue` |

### UI → Runtime (Manual)

These commands must be explicitly sent:

```rust
// Load files
controller.load_graph(path).await?;
controller.load_protein(path).await?;

// Control optimization
controller.start_optimization(config).await?;
controller.pause().await?;
controller.resume().await?;
controller.stop().await?;

// Set parameters
controller.set_parameter(key, value).await?;

// Shutdown
controller.shutdown().await?;
```

## Complete Example

See `/mnt/c/Users/Predator/Desktop/PRISM/crates/prism/examples/reactive_tui.rs` for a complete working example.

## Performance Considerations

### Event Processing Limit

The controller processes at most `max_events_per_poll` events per frame to prevent UI starvation:

```rust
// Default: 50 events per poll
// If runtime produces 1000 events/sec and UI runs at 60fps:
//   - Each frame processes ~16 events
//   - Latency: ~16ms (acceptable for visualization)

// For low-latency scenarios:
ReactiveControllerBuilder::new()
    .max_events_per_poll(200)  // Process more events per frame
    .build(...)
```

### Lagging Detection

If the UI falls behind, the controller logs warnings:

```
WARN: UI lagging behind runtime: skipped 127 events
```

This indicates you should either:
- Increase `max_events_per_poll`
- Reduce runtime event publishing rate
- Optimize UI rendering

### Zero-Copy Access

For high-frequency data visualization, use direct state access instead of events:

```rust
// Instead of waiting for events:
let history = controller.get_convergence_history();

// This reads directly from the lock-free ring buffer
// without creating event objects
```

## Error Handling

The reactive controller uses `anyhow::Result` with context:

```rust
match controller.poll_events(app) {
    Ok(()) => {}
    Err(e) => {
        eprintln!("Event processing error: {:?}", e);
        // Decide whether to continue or shutdown
    }
}

// Commands can also fail:
if let Err(e) = controller.start_optimization(config).await {
    app.dialogue.add_system_message(&format!("Failed to start: {}", e));
}
```

## Testing

### Unit Tests

```rust
#[tokio::test]
async fn test_controller_event_handling() {
    let event_bus = EventBus::new(16);
    let state = Arc::new(StateStore::new(100));
    let (cmd_tx, _) = mpsc::channel(16);

    let event_rx = event_bus.subscribe();
    let mut controller = ReactiveController::new(event_rx, state, cmd_tx);
    let mut app = App::new(None, "coloring".into(), 0)?;

    // Publish test event
    event_bus.publish(PrismEvent::GraphLoaded {
        vertices: 500,
        edges: 12500,
        density: 0.1,
        estimated_chromatic: 48,
    }).await?;

    // Poll and verify
    controller.poll_events(&mut app)?;
    assert_eq!(app.optimization.max_iterations, 48000);
}
```

### Integration Tests

See `/mnt/c/Users/Predator/Desktop/PRISM/crates/prism/tests/reactive_integration.rs`

## Troubleshooting

### Events not updating UI

**Problem**: UI doesn't reflect runtime state changes.

**Solutions**:
1. Ensure `controller.poll_events(app)` is called every frame
2. Check that runtime is publishing events: `event_bus.subscriber_count()`
3. Verify no errors in event handling: enable debug logging

### UI freezing

**Problem**: UI becomes unresponsive during heavy computation.

**Solutions**:
1. Ensure runtime actors are on separate tokio tasks
2. Reduce `max_events_per_poll` to give more CPU to rendering
3. Use `tokio::time::sleep()` in actor loops to yield

### Commands not executing

**Problem**: User actions don't trigger runtime responses.

**Solutions**:
1. Check command channel capacity: increase if needed
2. Ensure runtime has a PipelineActor listening for commands
3. Add logging to command sends: `controller.load_graph()` should log

## See Also

- [Runtime Architecture](/mnt/c/Users/Predator/Desktop/PRISM/crates/prism/src/runtime/README.md)
- [Event System](/mnt/c/Users/Predator/Desktop/PRISM/crates/prism/src/runtime/events.rs)
- [App State](/mnt/c/Users/Predator/Desktop/PRISM/crates/prism/src/ui/app.rs)
