# Code Snapshot - Key Modified Files

These are the primary source files involved in the Cryo-UV pipeline.

## Files

| File | Size | Description |
|------|------|-------------|
| `fused_engine.rs` | 93 KB | Main GPU-accelerated MD+NHS engine |
| `nhs_amber_fused.cu` | 64 KB | CUDA kernel for fused simulation |
| `nhs_adaptive.rs` | 38 KB | Adaptive ensemble CLI binary |
| `uv_bias.rs` | 28 KB | UV pump-probe system |
| `neuromorphic.rs` | 15 KB | LIF neuron network |
| `pipeline.rs` | 12 KB | Pipeline orchestration |
| `nhs_diagnose.rs` | 7 KB | Diagnostic tool |

## Key Modifications Made

### fused_engine.rs

**Line 989-990**: Base friction increase
```rust
dt: 0.002,          // 2 fs timestep
gamma_base: 10.0,   // Base friction at 300K (ps^-1) - increased from 1.0
```

**Lines 1076-1108**: Equilibration friction boost
```rust
fn compute_cryo_friction(&self, temperature: f32) -> f32 {
    const EQUILIBRATION_STEPS: i32 = 10000;
    const EQUILIBRATION_GAMMA: f32 = 1000.0;  // Extreme damping

    let base_gamma = if self.timestep < EQUILIBRATION_STEPS {
        let progress = self.timestep as f32 / EQUILIBRATION_STEPS as f32;
        let decay = (-3.0 * progress).exp();
        EQUILIBRATION_GAMMA * decay + self.gamma_base * (1.0 - decay)
    } else {
        self.gamma_base
    };
    // ... temperature scaling
}
```

### nhs_amber_fused.cu

**Line 54**: Lowered LIF threshold
```cuda
#define LIF_THRESHOLD 0.5f  // Was 5.0f - too high for signal levels
```

**Lines 493-511**: Tuned signal processing
```cuda
// Lower noise floor for sensitivity
const float NOISE_FLOOR = 0.002f;  // Was 0.005f

// Increased amplification
float bidirectional_signal = (abs_change - NOISE_FLOOR) * 10.0f;  // Was 5.0f

// Lower exclusion threshold
const float EXCLUSION_NOISE_FLOOR = 0.005f;  // Was 0.01f

// Increased exclusion weight
float exclusion_signal = (density_deviation - EXCLUSION_NOISE_FLOOR) * 5.0f;  // Was 2.0f

// Slower decay for accumulation
float effective_tau = tau_mem * 1.5f;  // Was 0.5f
```

**Lines 945-990**: Force and velocity clamping
```cuda
const float MAX_FORCE = 1000.0f;
// ... force clamping logic ...

const float MAX_VELOCITY = 100.0f;
// ... velocity clamping logic ...
```

## Diff Summary

```
fused_engine.rs:
  - gamma_base: 1.0 → 10.0
  + equilibration friction boost (1000 → 10 over 10k steps)

nhs_amber_fused.cu:
  - LIF_THRESHOLD: 5.0 → 0.5
  - NOISE_FLOOR: 0.005 → 0.002
  - Signal gain: 5× → 10×
  - EXCLUSION_NOISE_FLOOR: 0.01 → 0.005
  - Exclusion gain: 2× → 5×
  - Tau decay: 0.5× → 1.5×
  + Force clamping (MAX=1000)
  + Velocity clamping (MAX=100)
```

## Line Counts

```
$ wc -l *.rs *.cu
   2489 fused_engine.rs
    386 neuromorphic.rs
   1120 nhs_adaptive.rs
    213 nhs_diagnose.rs
    363 pipeline.rs
    858 uv_bias.rs
   1947 nhs_amber_fused.cu
   7376 total
```
