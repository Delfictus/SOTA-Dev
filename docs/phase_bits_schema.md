# `phase_bits` Schema (Gate G3)

**Field:** `phase_bits` (`u32`, per-spike) in
`<prefix>.topology.spike_events.arrow` and on
`crate::fused_engine::GpuSpikeEvent::phase_bits`.

**Authoritative decoder:** `crate::phase_bits::decode_phase_bits` in
`crates/prism-nhs/src/phase_bits.rs`. Unit tests in the same module
pin the layout against drift.

---

## Bit Layout (current implementation, schema version 1)

```
 31                                10 9                 0
 ┌─────────────────────────────────┬──┴─────────────────┐
 │     bits 31..10 — RESERVED      │  bits 9..0 — CCNS  │
 │     (unused; read-as-zero)      │  phase index 0–1023│
 └─────────────────────────────────┴────────────────────┘
 RESERVED_MASK = 0xFFFFFC00          CCNS_PHASE_INDEX_MASK
                                     = 0x000003FF
```

| Bits | Field | Type | Range | Meaning |
|---|---|---|---|---|
| **0–9** | `phase_index` | `u10` | `[0, 1023]` | Quantized CCNS protocol phase angle |
| **10–31** | `reserved` | `u22` | `0` (in current schema) | Unused; reserved for future extensions |

## Recovering the continuous phase angle

```
phi_radians = (phase_index / 1024.0) * 2 * PI
```

The denominator is **1024** (not 1023) because the angle wraps cyclically:
index 1024 ≡ index 0 ≡ phase 0. Pinned by:

* Engine emit:
  `crates/prism-nhs/src/protocol_state.rs:138`
  `pub current_phase_bits: u32  // 10-bit CCNS phase angle (0-1023), updated by Director each step`
* Engine consumer:
  `crates/prism-nhs/src/bin/nhs_rt_full.rs:4559`
  `let phi = (spike.phase_bits as f64 / 1024.0) * two_pi;`

## Authoritative producer

`crate::protocol_state::ProtocolState::current_phase_bits` is set by
the engine's protocol-state Director each MD step. The Director's
quantization rule: continuous CCNS phase angle in `[0, 2π)` is binned
into 1024 equal-width buckets; `phase_index = floor((phi / 2π) * 1024)
mod 1024`. This same value is then copied into every spike's
`GpuSpikeEvent::phase_bits` field at spike-emission time, so all
spikes within a single MD step share the same `phase_bits` value.

## What is **NOT** in `phase_bits` (frequently confused)

The following CCNS- and SDST-related fields are sometimes assumed to
be packed into `phase_bits` but are emitted separately:

| Concept | Where it actually lives | Type / shape |
|---|---|---|
| **CCNS 4-class phase** (cold_hold / ramp / warm_hold / cooling) | Separate `ccns_phase: u8` Arrow column (column 24 of `spike_events.arrow`) | `u8` enum: `0=cold_hold, 1=ramp, 2=warm_hold, 3=cooling` |
| **SDST hysteresis-phase fractions** | `binding_sites.json sites[].phase{}` per-site sub-dict | `{sdst_cold_fraction: f64, sdst_hot_fraction: f64, sdst_delta: f64}` (per-site only, not per-spike) |
| **Protocol stage flags** | `crate::protocol_state::ProtocolState` runtime struct + audit sidecars (`<prefix>.topology.asc_events.bin`, `<prefix>.topology.acl_contrast.bin`) | binary structures, not packed into spike fields |
| **Spike CCNS phase in the alternate u8 form** | Same as above (separate `ccns_phase: u8` column) | The 10-bit `phase_bits` index and the 4-class `ccns_phase: u8` are derived from the same underlying CCNS state but expose it at different resolutions |

If a future schema version packs SDST hot/cold derivative bits or
protocol-stage flags into `phase_bits` bits 10–31, this document
**MUST** be updated in the same commit as the producer change, and
the `crate::phase_bits` module's `RESERVED_MASK` constant + named
field accessors **MUST** be updated to expose the new sub-fields.

## Decoder pseudocode (any consumer language)

The decoder is one bit-and and one division. Reference Python:

```python
def decode_phase_bits(raw: int) -> dict:
    PHASE_MASK = 0x3FF
    PHASE_PERIOD = 1024
    phase_index = raw & PHASE_MASK
    reserved = (raw & ~PHASE_MASK) >> 10
    phase_radians = (phase_index / PHASE_PERIOD) * 2.0 * 3.141592653589793
    return {
        "phase_index": phase_index,
        "phase_radians": phase_radians,
        "reserved": reserved,
    }
```

Reference C:

```c
typedef struct {
    uint16_t phase_index;     // 0..1023
    double   phase_radians;   // 0..2*PI
    uint32_t reserved;        // 0 in current schema
} phase_bits_decoded_t;

phase_bits_decoded_t decode_phase_bits(uint32_t raw) {
    phase_bits_decoded_t out;
    out.phase_index   = (uint16_t)(raw & 0x3FFu);
    out.reserved      = (raw & 0xFFFFFC00u) >> 10;
    out.phase_radians = (double)out.phase_index / 1024.0 * 6.283185307179586;
    return out;
}
```

The Rust canonical decoder (`crate::phase_bits::decode_phase_bits`)
matches both above bit-for-bit and is the authoritative source for
PRISM-4D internal consumers.

## Schema-drift detection

Producer-side and consumer-side schemas agree iff:

1. `crate::phase_bits::CCNS_PHASE_INDEX_BITS == 10`
2. `crate::phase_bits::CCNS_PHASE_INDEX_MASK == 0x000003FF`
3. `crate::phase_bits::RESERVED_MASK == 0xFFFFFC00`
4. `crate::phase_bits::CCNS_PHASE_PERIOD == 1024`
5. The above unit tests in `crate::phase_bits::tests` all pass.
6. Engine-side denominator `1024` at
   `nhs_rt_full.rs:4559` matches `CCNS_PHASE_PERIOD`.

A failure of any of (1–6) is a schema-drift bug; this document and
the decoder module must be updated together.

## Schema version history

| Version | Date | Change |
|---|---|---|
| 1 | 2026-04-28 | Initial documentation and decoder module (Gate G3). Layout: bits 0–9 = CCNS phase index, bits 10–31 reserved. |
