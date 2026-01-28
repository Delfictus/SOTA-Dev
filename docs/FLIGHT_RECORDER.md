# PRISM-Zero Flight Recorder (PZFR) - User Guide

## Overview

The Flight Recorder provides real-time monitoring of physics engine telemetry without impacting simulation performance. Built on a lock-free ring buffer architecture, it achieves <5ns recording overhead in the hot path.

## Architecture

```text
┌──────────────────┐         ┌──────────────────┐
│  Physics Engine  │────────▶│  Telemetry Ring  │
│   (Hot Path)     │  <5ns   │   (100K frames)  │
└──────────────────┘         └─────────┬────────┘
                                       │
                                       │ async drain
                                       ▼
                             ┌──────────────────┐
                             │  prism-monitor   │
                             │  TUI Dashboard   │
                             │    (60 FPS)      │
                             └──────────────────┘
```

## Building

```bash
# Build the Flight Recorder dashboard
cargo build -p prism-core --bin prism-monitor --features telemetry

# Build in release mode for production use
cargo build -p prism-core --bin prism-monitor --features telemetry --release
```

## Running the Dashboard

```bash
# Start the live monitoring dashboard
cargo run -p prism-core --bin prism-monitor --features telemetry
```

### Dashboard Controls

- **SPACE**: Pause/resume display updates
- **q** or **ESC**: Quit dashboard
- **h**: Show help overlay

## Dashboard Layout

```text
┌─ PRISM-Zero Flight Recorder ─────────────────────────┐
│ Step: 1,234,567  │  Energy: -1,234.56 kcal/mol      │
│ Time: 12m 34s    │  Temp: 300.15 K                  │
│ Rate: 2.1 ms/cyc │  Accept: 85.3%                   │
├──────────────────┼──────────────────────────────────┤
│     Energy       │          Temperature              │
│ (convergence)    │        (thermostat)               │
├──────────────────┼──────────────────────────────────┤
│   Acceptance     │          Gradient                 │
│  (PIMC tuning)   │      (NLNM convergence)           │
└──────────────────────────────────────────────────────┘
```

## Integration with Physics Engine

### 1. Initialize Telemetry at Startup

```rust
use prism_core::telemetry;

fn main() {
    // Initialize the telemetry ring buffer once at startup
    telemetry::init_telemetry();

    // Run your physics simulation
    run_simulation();
}
```

### 2. Record Telemetry in Hot Loop

```rust
use prism_core::telemetry;
use std::time::Instant;

fn physics_step(step: u64, start_time: Instant, state: &SimulationState) {
    // Your physics calculations here
    let energy = calculate_hamiltonian(state);
    let temp = thermostat.temperature();
    let acceptance = monte_carlo.acceptance_rate();
    let gradient = calculate_gradient_norm(state);

    // Record telemetry (<5ns overhead)
    telemetry::record_simulation_state(
        step,
        start_time,
        energy,
        temp,
        acceptance,
        gradient,
    );
}
```

### 3. Optional: Configure Sampling Rate

```rust
use prism_core::telemetry::{self, TelemetryConfig};

// Record every 10th frame for reduced overhead
let config = TelemetryConfig {
    high_frequency: true,
    sample_rate: 10,
    monitor_gradients: true,
};

telemetry::configure(config);
```

## Performance Characteristics

- **Recording Overhead**: <5 nanoseconds per frame
- **Memory Usage**: 3.2 MB (100,000 frames × 32 bytes)
- **Buffer Capacity**: 100,000 frames (~13 hours at 500ms per cycle)
- **Dashboard Refresh**: 60 FPS (16ms)
- **SIMD Alignment**: 32-byte aligned frames for cache efficiency

## Telemetry Metrics

### Energy (kcal/mol)
Hamiltonian energy showing system convergence. Expect gradual stabilization as simulation equilibrates.

### Temperature (K)
Thermostat temperature control. Should oscillate around target (e.g., 300K) with proper thermostat tuning.

### Acceptance Rate (%)
Monte Carlo acceptance rate for Path Integral Monte Carlo (PIMC). Target: 70-90% for efficient sampling.

### Gradient Norm
Magnitude of energy gradient for Non-Linear Normal Mode (NLNM) analysis. Converges toward zero at equilibrium.

## Troubleshooting

### Dashboard shows "No data available"

The physics engine hasn't started recording telemetry yet. Ensure:
1. `telemetry::init_telemetry()` was called
2. `telemetry::record_simulation_state()` is being called in the physics loop

### Buffer overflow warnings

The physics engine is producing data faster than the dashboard can consume. Options:
1. Increase sampling rate (record every Nth frame)
2. Reduce dashboard update frequency
3. Run dashboard on separate machine via network export

### High CPU usage

The dashboard runs at 60 FPS by default. To reduce CPU:
1. Lower refresh rate in `REFRESH_RATE` constant
2. Use `--release` build for optimized performance

## Advanced Usage

### Exporting Telemetry Data

For post-simulation analysis, drain frames programmatically:

```rust
use prism_core::telemetry;

fn export_telemetry() {
    let frames = telemetry::drain_frames();

    // Export to CSV, Parquet, or other format
    for frame in frames {
        println!("{},{},{},{},{}",
            frame.step,
            frame.timestamp_ns,
            frame.energy,
            frame.temperature,
            frame.acceptance_rate
        );
    }
}
```

### Network Monitoring

For distributed simulations, export telemetry over the network:

```rust
// Producer (physics engine)
let frames = telemetry::drain_frames();
send_to_dashboard_server(frames);

// Consumer (remote dashboard)
receive_and_display(frames);
```

## Architecture Details

### Lock-Free Ring Buffer
- Uses `crossbeam-queue::ArrayQueue` for zero-contention producer/consumer
- Atomic operations ensure thread-safe access without locks
- Drop policy: new frames are dropped when buffer is full (never block physics)

### SIMD Optimization
- 32-byte aligned `TelemetryFrame` struct
- Fits exactly in L1 cache line (64 bytes = 2 frames)
- Enables efficient vectorized operations

### Zero-Allocation Hot Path
- Pre-allocated ring buffer at initialization
- Fixed-size structs avoid heap allocation
- `#[inline(always)]` for recording functions eliminates call overhead

## Future Enhancements

- [ ] Apache Arrow export for columnar analytics
- [ ] Network streaming via TCP/UDP
- [ ] GPU telemetry integration (CUDA kernel metrics)
- [ ] Alert system for anomaly detection
- [ ] Historical playback and scrubbing

## See Also

- `crates/prism-core/src/telemetry.rs` - Core telemetry infrastructure
- `crates/prism-core/src/bin/prism-monitor.rs` - TUI dashboard implementation
- PRISM GPU Plan §4: Real-Time Telemetry Architecture
