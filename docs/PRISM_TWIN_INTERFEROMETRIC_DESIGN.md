# PRISM-TWIN: Interferometric Spike Dynamics — Redesigned Architecture

## Core Concept

Two MD simulations are not "scout and observer." They are two COMPLEMENTARY 
PROBES of the same conformational landscape — like two eyes creating DEPTH 
PERCEPTION that neither eye has alone.

The value is not in individual spike trains. The value is in the 
CROSS-CORRELATION BETWEEN THEM.

## Five Layers of Coupling

### Layer 1: Spike Density Exchange (spatial)
Share WHERE spikes occur → adjust thresholds.
Information: spatial location of active regions.

### Layer 2: Temporal Cross-Correlation (temporal)
For every voxel, compute CCF between A's and B's spike trains.
```
CCF(τ) = Σ spike_A(t) × spike_B(t + τ) / norm
```
Reveals: reproducibility, propagation speed, barrier height, mechanism.

### Layer 3: Oscillator Frequency Coupling (spectral) [FUTURE v2]
A's oscillators shift resonant frequency based on B's dominant spike frequency.
Adaptive frequency matching — oscillators learn each other's conformational dynamics.

### Layer 4: Perturbation Response Differential (mechanical)
Transfer function: H(ω) = response_B(ω) / (response_A(ω) + NMA_input(ω))
Per-residue mechanical susceptibility map.

### Layer 5: Causal Spike Propagation (information-theoretic) [FUTURE v2]
Transfer entropy TE(A→B) reveals causal flow between simulations.
Separates deterministic conformational events from stochastic noise.

## Implementation Phases

### Phase 1 (Gates 1-4): Layers 1 + 2
- Dual streams, interleaved stepping
- Per-voxel paired spike history ring buffers
- Real-time CCF computation on GPU
- Spike density exchange for threshold adaptation
- Output: CCF profiles per residue, consensus/differential classification

### Phase 2 (Gates 5-8): Layer 4
- NMA perturbation in stream B only
- Transfer function computation in frequency domain
- Mechanical susceptibility map per residue
- Output: barrier classification (elastic/cooperative/rigid)

### Phase 3 (Future v2): Layers 3 + 5
- Adaptive oscillator frequency coupling
- Transfer entropy on paired spike trains
- Causal flow analysis
- Patent claims filed NOW, implementation after 10K run

## GPU Architecture: Interleaved Stepping

```
for step in 0..total_steps:
    // Advance both streams by one fused step
    engine_a.step()    // on stream_a
    engine_b.step()    // on stream_b (concurrent on GPU)
    
    // Every exchange_interval steps:
    if step % exchange_interval == 0:
        // 1. Compute spike density (Layer 1)
        compute_spike_density(buffer_a, density_a)  // stream_exchange
        compute_spike_density(buffer_b, density_b)
        
        // 2. Compute cross-correlation (Layer 2)
        compute_ccf(ring_a, ring_b, ccf_map)        // stream_exchange
        
        // 3. Apply threshold modifications (Layer 1)
        apply_threshold_mod(density_b → thresholds_a) // stream_a
        apply_threshold_mod(density_a → thresholds_b) // stream_b
        
        // 4. Update ring buffers
        rotate_ring_buffers(ring_a, ring_b)
```

## Data Structures

### Per-Voxel Ring Buffer (for CCF computation)
```
spike_ring_a[n_voxels][RING_SIZE]  // recent spike history from A
spike_ring_b[n_voxels][RING_SIZE]  // recent spike history from B
ring_head[n_voxels]                 // current write position

RING_SIZE = 256  // captures ~500 steps of spike history
                 // enough for CCF at τ = [-128, +128] steps
```

### Per-Voxel CCF Output
```
ccf_map[n_voxels][CCF_LAGS]  // cross-correlation at each lag
CCF_LAGS = 64                 // τ from -32 to +31 steps

// Derived per-voxel features:
ccf_peak_lag       // lag at max CCF → propagation delay
ccf_peak_value     // max CCF → reproducibility score
ccf_width          // FWHM → transition sharpness
ccf_asymmetry      // (CCF(τ>0) - CCF(τ<0)) → causal direction
```

### Per-Residue Output Features (~50 new)
```
// Consensus (12):
spike_agreement_ratio, consensus_intensity_mean,
consensus_phase_profile[5], consensus_spatial_coherence,
consensus_temporal_onset, n_consensus_neighbors

// Cross-correlation (12):
ccf_peak_lag, ccf_peak_value, ccf_width, ccf_asymmetry,
ccf_per_phase[5], ccf_frequency_peak, ccf_reproducibility,
ccf_lag_consistency

// Differential (18):
b_over_a_spike_ratio, nma_exclusive_count,
thermal_exclusive_count, b_over_a_intensity_ratio,
nma_responsive_mode, nma_mode_eigenvalue,
barrier_classification, per_phase_differential[5],
differential_onset_lag, nma_work_at_residue,
mechanical_sensitivity, susceptibility_magnitude

// Scout/propagation (8):
scout_lead_time, scout_predictive_value,
phase_offset_enrichment, scout_intensity_at_onset,
scout_spatial_propagation, mutual_information,
transfer_entropy_a_to_b, causal_flow_direction
```

## Patent Claims (file immediately)

1. Coupled observation (not force coupling) between parallel MD simulations
2. Spike-density-driven adaptive detector sensitivity across simulations
3. Cross-correlation function on neuromorphic spike trains from parallel MD
4. Differential perturbation for activation energy measurement
5. Adaptive oscillator frequency matching between parallel simulations
6. Transfer entropy on paired MD spike trains for causal mechanism discovery
7. Interferometric spike dynamics for conformational landscape depth perception
