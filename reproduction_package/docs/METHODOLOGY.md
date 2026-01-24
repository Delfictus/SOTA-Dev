# PRISM4D Cryo-UV Pump-Probe Methodology

## Scientific Basis

### The Cryptic Site Problem

Cryptic binding sites are protein cavities that:
- Are **closed** in crystal structures (no visible pocket)
- **Open transiently** during dynamics
- **Bind ligands** when open
- Are **invisible** to traditional docking

Standard MD simulations often miss cryptic sites because:
1. Timescales are too short (ns vs μs-ms opening)
2. Energy barriers prevent spontaneous opening
3. No directed perturbation to trigger opening

### The Cryo-UV Solution

We combine two physical principles:

#### 1. Cryogenic Contrast Enhancement

At low temperatures (100K):
- Water approaches discrete freeze transition
- Protein fluctuations are suppressed
- Aromatic residues become "sluggish"
- Signal-to-noise ratio increases dramatically

#### 2. UV Pump-Probe Excitation

Aromatic residues (TRP, TYR, PHE) absorb at 280nm:
- UV burst deposits localized energy
- Aromatic ring vibrates/rotates
- Steric gate opens
- Water is displaced (dewetting)

### Combined Protocol

```
Time ───────────────────────────────────────────────────────────►

          FREEZE                PUMP-PROBE              VALIDATE
     ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
     │   100K      │        │  100K→300K  │        │    300K     │
     │   Survey    │   →    │  UV Bursts  │   →    │   Confirm   │
     │   Baseline  │        │  Trigger    │        │   Opening   │
     └─────────────┘        └─────────────┘        └─────────────┘
           ↓                      ↓                      ↓
     Map hydrophobic       Detect dewetting         Validate that
     surface in frozen     as spikes during         pockets remain
     reference state       temperature ramp         open at 300K
```

## Implementation Details

### 1. Structure Preparation

```bash
prism-prep input.pdb output.json --use-amber --mode cryptic --strict
```

Steps:
1. Chain contact analysis (identify interfaces)
2. Glycan handling (strip for cryptic mode)
3. AMBER reduce for H-bond optimization
4. PDBFixer for missing atoms/residues
5. AMBER ff14SB topology generation
6. Validation (bonds, angles, clashes)

### 2. Cryo-UV Simulation

#### Langevin Dynamics

The Langevin equation:
```
m·dv/dt = F - γ·v + √(2·γ·kT/m)·R(t)
```

Where:
- `F` = AMBER ff14SB forces
- `γ` = friction coefficient (10 ps⁻¹ base)
- `R(t)` = Gaussian random force

#### Equilibration Protocol

Critical fix for numerical stability:
```
γ(t) = 1000·exp(-3·t/10000) + 10·(1 - exp(-3·t/10000))
```

This provides:
- Extreme damping (1000 ps⁻¹) initially
- Exponential decay to base (10 ps⁻¹)
- Prevents velocity explosion from unminimized structures

#### Temperature Protocol

```
T(step) = {
    100K                           if step < survey_steps
    100K + (300K-100K)·(step-survey)/(ramp_steps)   during ramp
    300K                           after ramp
}
```

Default values:
- Survey: 50,000 steps (100 ps at 2fs timestep)
- Ramp: 50,000 steps
- Warm: 50,000 steps

### 3. Water Inference

Water density is inferred from the exclusion field:
```
ρ_water(x) = ρ_bulk × (1 - E(x)) × P(x) × √(T/300)
```

Where:
- `ρ_bulk` = 0.0334 molecules/Å³ (bulk water)
- `E(x)` = exclusion field (0-1, from atom overlap)
- `P(x)` = polar enhancement (1 + 0.3·polar_field)
- `T` = temperature (thermal factor)

### 4. Neuromorphic Spike Detection

#### LIF Neuron Model

```
V(t+1) = V(t)·exp(-dt/τ_eff) + I(t)

where:
  τ_eff = 1.5·τ_mem  (slower decay for accumulation)
  I(t) = bidirectional_signal + exclusion_signal
```

#### Signal Processing

```
ΔDensity = ρ_water(t) - ρ_water(t-1)

if |ΔDensity| > 0.002:  # noise floor
    bidirectional_signal = (|ΔDensity| - 0.002) × 10

if |ρ_water - ρ_bulk| > 0.005:  # exclusion floor
    exclusion_signal = (|ρ_water - ρ_bulk| - 0.005) × 5

combined_signal = bidirectional + exclusion
```

#### Spike Generation

```
if V > 0.5:  # threshold
    emit SPIKE at (x, y, z)
    V = 0  # reset
```

### 5. Correlation Analysis

#### RMSF Calculation

From ensemble of N frames:
```
RMSF(i) = √( (1/N) Σ_t |r_i(t) - <r_i>|² )
```

With Kabsch alignment to remove rigid-body motion.

#### Spike-RMSF Correlation

For each high-RMSF residue (> mean + σ):
1. Find nearest spike hotspot within 10Å
2. Compute combined score: `RMSF × spike_count`
3. Rank by combined score

Sites with high combined scores are **druggable cryptic targets**.

## Parameter Reference

### Critical Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gamma_base` | 10.0 ps⁻¹ | Sufficient damping without overdamping |
| `LIF_THRESHOLD` | 0.5 | Tuned for cryo-UV signal levels |
| `NOISE_FLOOR` | 0.002 | Below thermal fluctuation (~0.001) |
| `Signal amplification` | 10× | Compensates for small density changes |
| `τ_decay` | 1.5× | Allows signal accumulation |
| `Correlation radius` | 10 Å | Captures allosteric effects |

### Physics Constants

| Constant | Value | Source |
|----------|-------|--------|
| `ρ_bulk` | 0.0334 mol/Å³ | Bulk water at 300K |
| `ε_water` | 78.5 | Dielectric at 300K |
| `ε_ice` | 3.2 | Dielectric at 100K |
| `UV_λ` | 280 nm | Aromatic absorption peak |

## Validation

### Expected Results

For a valid cryptic site:
1. **RMSF > 10 Å** at the site residues
2. **Spike count > 100** in the region
3. **Aromatic residue within 8 Å** of hotspot
4. **Combined score > 1000**

### Negative Controls

1. **Buried hydrophobic core**: High exclusion, low RMSF
2. **Surface loops**: High RMSF, low spikes (already solvated)
3. **Random position**: Low correlation

### Positive Controls

1. **Known cryptic sites** from CryptoBench dataset
2. **Apo-holo pairs** where holo shows pocket

## Troubleshooting

### No Spikes Detected

1. Check `LIF_THRESHOLD` (should be ~0.5)
2. Verify temperature protocol is running (not constant 300K)
3. Check water density is varying (not stuck at 0)

### Simulation Explodes

1. Increase equilibration friction (`EQUILIBRATION_GAMMA`)
2. Enable force clamping (`MAX_FORCE = 1000`)
3. Check for bad contacts in input structure

### High RMSF But No Correlation

1. Increase correlation radius (try 15 Å)
2. Lower RMSF threshold
3. Check coordinate frames match (both uncentered)

## References

1. Cimermancic et al. (2016) CryptoSite. *J Mol Biol*
2. Beglov et al. (2018) FTMap. *J Med Chem*
3. Bowman & Geissler (2012) Cryptic binding sites. *PNAS*
