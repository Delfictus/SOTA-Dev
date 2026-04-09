# PRISM-TWIN v3.0: Fully Autonomous CUDA Graph Architecture

**Status**: IMPLEMENTATION PLAN  
**Date**: 2026-04-09  
**Prerequisite**: 36 commits on feat/twin-multistream (Phase A+B complete, prism-cuda-ext crate built)  

---

## The Problem

CUDA graphs replay kernel launches with BAKED-IN scalar parameters.
The PRISM engine changes behavior across steps:
- Temperature ramps through 5 CCNS phases (100K→300K→100K)
- UV bursts fire at intervals
- Adaptive bias updates periodically
- Protocol timing depends on `--fast` vs standard

If we capture one step and replay 40M times, the simulation runs at
Step 0's temperature FOREVER. The science is destroyed.

## The Solution: VRAM-Driven Thermostat + Director Kernel

### Component 1: ProtocolState struct in VRAM

```cuda
struct ProtocolState {
    // Step tracking
    uint32_t current_step;           // incremented by Director each iteration
    uint32_t total_steps;            // simulation length

    // Temperature (CCNS 5-phase)
    float current_temperature;       // computed by Director based on phase
    float start_temp;                // cold temperature (e.g., 100K)
    float end_temp;                  // warm temperature (e.g., 300K)

    // Phase boundaries (in AMBER steps, set once at init)
    int32_t cold_hold_end;           // = cold_hold_steps
    int32_t ramp_end;               // = cold_hold_steps + ramp_steps
    int32_t warm_hold_end;          // = ramp_end + warm_hold_steps
    int32_t ramp_down_end;          // = warm_hold_end + ramp_down_steps
    // cold_return: everything after ramp_down_end

    // UV burst state
    int32_t uv_burst_active;        // 0 or 1
    float uv_burst_energy;           // kcal/mol
    float uv_wavelength_nm;          // current wavelength
    int32_t uv_burst_interval;       // steps between bursts
    int32_t uv_burst_duration;       // steps per burst
    int32_t uv_target_idx;           // current aromatic target

    // Langevin parameters
    float dt;                        // timestep
    float gamma;                     // friction coefficient
    float cutoff;                    // nonbonded cutoff

    // Fused step tracking
    int32_t fused_inner_steps;       // --fused-steps value
    int32_t fused_step_counter;      // 0..fused_inner_steps within each outer step
};
```

### Component 2: Director Kernel

```cuda
extern "C" __global__ void protocol_director(ProtocolState* state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Increment step
    state->current_step += 1;
    uint32_t step = state->current_step;

    // ── CCNS 5-Phase Temperature State Machine ──
    if (step < state->cold_hold_end) {
        state->current_temperature = state->start_temp;
    } else if (step < state->ramp_end) {
        float progress = (float)(step - state->cold_hold_end)
                       / (float)(state->ramp_end - state->cold_hold_end);
        state->current_temperature = state->start_temp
                                   + progress * (state->end_temp - state->start_temp);
    } else if (step < state->warm_hold_end) {
        state->current_temperature = state->end_temp;
    } else if (step < state->ramp_down_end) {
        float progress = (float)(step - state->warm_hold_end)
                       / (float)(state->ramp_down_end - state->warm_hold_end);
        state->current_temperature = state->end_temp
                                   - progress * (state->end_temp - state->start_temp);
    } else {
        state->current_temperature = state->start_temp; // cold_return
    }

    // ── UV Burst Scheduling ──
    int32_t burst_cycle = step % state->uv_burst_interval;
    state->uv_burst_active = (burst_cycle < state->uv_burst_duration) ? 1 : 0;

    // Wavelength hopping (cycle through scan wavelengths)
    // Simplified: rotate through 280, 275, 260 nm
    int32_t wl_idx = (step / state->uv_burst_interval) % 3;
    float wavelengths[3] = {280.0f, 275.0f, 260.0f};
    state->uv_wavelength_nm = wavelengths[wl_idx];

    // ── Fused step counter ──
    state->fused_step_counter = (state->fused_step_counter + 1)
                               % state->fused_inner_steps;
}
```

### Component 3: Physics Kernel Modification

**Current** (scalar parameters, baked into graph):
```cuda
extern "C" __global__ void nhs_amber_fused_step(
    ...,
    float temp_start,          // ← BAKED IN by graph
    float temp_end,            // ← BAKED IN
    int temp_ramp_steps,       // ← BAKED IN
    int temp_hold_steps,       // ← BAKED IN
    int temp_current_step,     // ← BAKED IN (always Step 0!)
    ...,
    int uv_burst_active,       // ← BAKED IN
    float uv_burst_energy,     // ← BAKED IN
    float uv_wavelength_nm,    // ← BAKED IN
    ...
) {
    float target_temp;
    if (temp_current_step < temp_ramp_steps) {
        float t = (float)temp_current_step / (float)temp_ramp_steps;
        target_temp = temp_start + t * (temp_end - temp_start);
    } else {
        target_temp = temp_end;
    }
    ...
}
```

**Modified** (reads from VRAM, graph-compatible):
```cuda
extern "C" __global__ void nhs_amber_fused_step(
    ...,
    const ProtocolState* __restrict__ d_protocol,  // REPLACES 5 temp scalars + UV scalars
    ...
) {
    // Temperature comes from Director kernel's computation
    float target_temp = d_protocol->current_temperature;

    // UV state comes from Director kernel
    int uv_burst_active = d_protocol->uv_burst_active;
    float uv_burst_energy = d_protocol->uv_burst_energy;
    float uv_wavelength_nm = d_protocol->uv_wavelength_nm;
    float dt = d_protocol->dt;
    float gamma = d_protocol->gamma;

    // REST OF PHYSICS IS IDENTICAL — same forces, same integration,
    // same thermostat math. Only the INPUT SOURCE changed.
    ...
}
```

**This is a parameter delivery refactor, NOT a physics change.**
The Langevin thermostat, force computation, SHAKE, GBSA — all identical.
Only the mechanism for delivering temperature/UV values to the kernel changes.

**CRITICAL PERFORMANCE NOTE**: Global memory reads (from ProtocolState*)
are slower than constant memory parameter reads. To avoid L2 cache round-trips
on every thread, read the struct into LOCAL REGISTERS once at kernel entry:

```cuda
extern "C" __global__ void nhs_amber_fused_step(
    ...,
    const ProtocolState* __restrict__ d_protocol,
    ...
) {
    // Load protocol state into registers ONCE (all threads read same values)
    // The __restrict__ hint + L1 cache means this is a single broadcast read
    // across all threads in the warp — effectively free after the first thread.
    const float target_temp = d_protocol->current_temperature;
    const int uv_active = d_protocol->uv_burst_active;
    const float uv_energy = d_protocol->uv_burst_energy;
    const float uv_wl = d_protocol->uv_wavelength_nm;
    const float dt = d_protocol->dt;
    const float gamma = d_protocol->gamma;
    const int timestep = d_protocol->current_step;

    // From this point forward, target_temp etc. are in registers — 
    // identical performance to the original scalar parameter path.
    // The compiler treats them the same way as __constant__ parameters
    // once they're in registers.

    // ... rest of kernel uses target_temp, uv_active, etc. from registers ...
}
```

This ensures ZERO performance regression from the parameter delivery change.
The first thread in each warp loads from L1/L2 (one cache line, ~80 bytes).
All subsequent threads in the warp get the same values via warp broadcast.
After the initial load, the register-based values are indistinguishable
from the original scalar parameter path in terms of ALU throughput.

---

## Gated Implementation Plan

### Gate 0: ProtocolState struct + Director kernel (Week 1)

**New file**: `crates/prism-gpu/src/kernels/protocol_director.cu`
- ProtocolState struct definition (shared header)
- `protocol_director` kernel: step increment + phase state machine + UV scheduling
- `init_protocol_state` kernel: initialize from Rust-side protocol config

**Build**: standard `compile_kernel` in build.rs

**Rust**: allocate `CudaSlice<u8>` for ProtocolState (sizeof = ~80 bytes),
initialize from CryoUvProtocol fields, upload once at simulation start.

**Gate criteria**:
- PTX compiles
- `cargo test -p prism-nhs --lib` passes 169/0/3
- Director kernel produces correct temperature ramp when tested standalone
  (initialize state, run Director 35000 times, read back temperatures,
  verify they match CryoUvProtocol::fast_35k() schedule)

**Verification**:
```bash
# Director kernel unit test:
# Initialize ProtocolState for fast_35k (cold_hold=14000, ramp=10500, warm=7000, ...)
# Run director 35000 times
# Download current_temperature at steps 0, 14000, 24500, 31500, 35000
# Verify: 100K, 100K→300K transition, 300K, 300K→100K transition, 100K
```

### Gate 1: Modify nhs_amber_fused_step to read from ProtocolState (Week 2)

**This is the FROZEN kernel modification.** Justification:
- Zero physics change — same math, same forces, same integration
- Only the parameter delivery mechanism changes
- Validated by running BENCH60 subset before/after and diffing DCC scores

**Changes to nhs_amber_fused.cu (lines 894-898, 932-942)**:
- Replace 5 temperature scalar params with `const ProtocolState*`
- Replace `uv_burst_active`, `uv_burst_energy`, `uv_wavelength_nm` params with reads from ProtocolState
- Remove internal temperature computation (lines 936-942) — Director kernel handles it
- Keep `dt`, `gamma`, `cutoff` as either ProtocolState fields or separate params

**Changes to fused_engine.rs**:
- Replace `.arg(&temp_start).arg(&temp_end).arg(&temp_ramp_steps)...` with `.arg(&self.d_protocol_state)`
- Add `d_protocol_state: CudaSlice<u8>` field
- Initialize from CryoUvProtocol at topology load time
- The Director kernel increments the state each step

**Changes to nhs_voxel_step_multi_lif**:
- Same pattern — read temperature from ProtocolState instead of scalar

**Gate criteria**:
- `cargo test -p prism-nhs --lib` passes
- 1btl E2E with `--fast --hysteresis --prism-therm --spike-percentile 95 --fused-steps 4 --hmr --adaptive-dt -v`
  produces sites within 1Å DCC of pre-modification baseline
- Spike counts within 5% of baseline (stochastic variation expected)

**Verification (CRITICAL)**:
```bash
# Run BEFORE modification:
scripts/prism-validate-and-run.sh -t 1btl.topology.json -o /tmp/baseline_1btl \
    --fast --hysteresis --spike-percentile 95 --prism-therm \
    --fused-steps 4 --hmr --adaptive-dt --replica-seed 42 -v

# Run AFTER modification:
scripts/prism-validate-and-run.sh -t 1btl.topology.json -o /tmp/modified_1btl \
    --fast --hysteresis --spike-percentile 95 --prism-therm \
    --fused-steps 4 --hmr --adaptive-dt --replica-seed 42 -v

# Compare:
python3 -c "
import json
b = json.load(open('/tmp/baseline_1btl/1btl.binding_sites.json'))
m = json.load(open('/tmp/modified_1btl/1btl.binding_sites.json'))
# Sites should be nearly identical (same seed, same physics)
for i, (bs, ms) in enumerate(zip(b['sites'], m['sites'])):
    dx = sum((a-b)**2 for a,b in zip(bs['centroid'], ms['centroid']))**0.5
    print(f'Site {i}: DCC delta = {dx:.2f} Å')
    assert dx < 1.0, f'REGRESSION: Site {i} moved {dx:.2f} Å'
"
```

### Gate 2: Stream capture of Director + Physics (Week 3)

Now that the physics kernel reads from VRAM instead of scalars, the
entire sequence is graph-capturable:

```
Director → Physics A → Physics B → Multi-LIF A → Multi-LIF B → LADD
→ Compact A → Compact B → Adapt B→A → Adapt A→B → Recovery
```

**All kernels read from device pointers** — no scalar values are
baked in. The Director updates ProtocolState before each physics step.

**Capture**:
1. `cuStreamBeginCapture(stream_exchange, RELAXED)`
2. Launch Director kernel
3. Launch engine_a.step() (captures fused + multi_lif + LADD)
4. Launch engine_b.step()
5. Launch compact_a + compact_b (device-side)
6. Launch adapt_B→A + adapt_A→B (ring buffer)
7. Launch recovery
8. `cuStreamEndCapture(stream_exchange, &captured_graph)`

**Wrap in WHILE conditional node**:
- Condition: `d_protocol_state->current_step < total_steps`
- Body: the captured graph above
- One `cuGraphLaunch` for the entire simulation

**Gate criteria**:
- Graph captures successfully (no STREAM_CAPTURE_ISOLATION errors)
- Graph launches and runs to completion
- Output matches host-mediated baseline within DCC tolerance

### Gate 3: Zero-copy exhaust + harvester thread (Week 4)

With the physics running autonomously inside the graph:
1. The compact kernel writes to the mapped host RAM exhaust buffer
2. A Rust background thread tails the exhaust buffer
3. Spike data streams to NVMe as Parquet (channel-tagged, timestamped)

**Gate criteria**:
- Parquet file produced with correct spike counts
- Channel tags (LIF=0, UV=1, RAF=2, EFP=3, LADD=4, COFIRE=5) preserved
- File size: ~2 GB Parquet vs ~22 GB JSON for 145M spikes

### Gate 4: Full validation on 3 targets (Week 5)

Run 1btl, 4obe, 1w50 with:
```bash
--coupled-twin --graph-coupling --multi-stream 8 \
--fast --hysteresis --spike-percentile 95 --prism-therm \
--fused-steps 4 --hmr --adaptive-dt -v
```

**Gate criteria**:
- All 3 targets complete without crash
- DCC within 1Å of host-mediated baseline
- Wall-clock faster than host-mediated (no kernel launch overhead)
- nvitop shows high SM util + high MBW (physics running, not spinning)
- Spike data preserved in Parquet with all channel tags

---

## What This Changes vs What Stays the Same

### MODIFIED (parameter delivery refactor):
- `nhs_amber_fused.cu`: temperature/UV params → ProtocolState* read
- `nhs_voxel_step_multi_lif`: same change
- `fused_engine.rs`: scalar arg chain → ProtocolState allocation + Director launch

### UNCHANGED:
- All force computation (bonds, angles, dihedrals, LJ, exclusions, 1-4)
- Velocity Verlet integration
- Langevin thermostat math
- SHAKE constraints
- Multi-LIF oscillator network (K=8 RAF)
- UV excited state dynamics
- EFP detection
- LADD departure detection
- Spike recording format
- All Python pipeline code

### NEW:
- `protocol_director.cu`: Director kernel + ProtocolState struct
- `d_protocol_state`: 80-byte device allocation
- Graph capture of full step sequence
- Parquet exhaust pipeline

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Physics regression from kernel change | Gate 1: DCC comparison before/after, same seed |
| Temperature ramp incorrect in Director | Gate 0: standalone Director test vs CryoUvProtocol schedule |
| Graph capture fails | Graceful fallback to host-mediated (existing code) |
| Parquet I/O too slow | Async harvester thread, batched writes |
| NMA perturbation not in Director | NMA kernel runs separately (already a device kernel), add to graph after physics |

## Why This Is The Correct Final Architecture

The VRAM-driven thermostat eliminates the LAST dependency between
the CPU and the GPU simulation loop. The Director kernel replaces
the CPU's role as protocol manager. The conditional WHILE graph
replaces the CPU's role as loop controller. The exhaust pipeline
replaces the CPU's role as data collector.

The CPU does exactly three things:
1. Build the graph (50ms, once)
2. Launch the graph (1μs, once)
3. Harvest spike data from mapped RAM (background thread, async)

The GPU does everything else for 5 minutes straight.
