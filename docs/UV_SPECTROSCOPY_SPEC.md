# UV Spectroscopy Enhancement Specification

## Current State (v1.3.0)
- Simple force perturbation on aromatic rings
- Single wavelength (280nm) scaling
- No true electronic state modeling

## Required Enhancements for Publication-Quality UV Probe

### 1. Multi-Wavelength Absorption Model

```rust
pub struct UvAbsorptionSpectrum {
    // Wavelength-dependent extinction coefficients
    pub wavelengths: Vec<f32>,           // [258, 274, 280] nm
    pub extinction: HashMap<String, Vec<f32>>,  // Per-residue spectrum
}

// Proper λmax values (Beer-Lambert law)
pub const UV_CHROMOPHORES: &[ChromophoreSpec] = &[
    ChromophoreSpec {
        residue: "TRP",
        lambda_max: 280.0,  // nm
        epsilon: 5600.0,    // M⁻¹cm⁻¹
        bandwidth: 15.0,    // nm (FWHM)
        transition: "π→π* (La band)"
    },
    ChromophoreSpec {
        residue: "TYR",
        lambda_max: 274.0,
        epsilon: 1400.0,
        bandwidth: 12.0,
        transition: "π→π* (phenol)"
    },
    ChromophoreSpec {
        residue: "PHE",
        lambda_max: 258.0,
        epsilon: 200.0,
        bandwidth: 10.0,
        transition: "π→π* (benzyl)"
    },
    ChromophoreSpec {
        residue: "CYS-CYS",  // Disulfide
        lambda_max: 250.0,
        epsilon: 300.0,
        bandwidth: 20.0,
        transition: "σ→σ* (S-S)"
    },
];
```

### 2. Energy Deposition → Local Temperature

```rust
/// UV photon absorption causes local heating
pub fn compute_local_heating(
    photon_energy: f32,      // eV (280nm ≈ 4.43 eV)
    absorption_cross: f32,   // Å²
    fluence: f32,            // photons/Å²
) -> f32 {
    // Energy deposited per chromophore
    let energy_deposited = photon_energy * absorption_cross * fluence;  // eV

    // Convert to temperature increase (equipartition)
    // ΔT = E / (3/2 * k_B * N_atoms)
    let n_ring_atoms = 6.0;  // Approximate
    let k_B = 8.617e-5;      // eV/K

    energy_deposited / (1.5 * k_B * n_ring_atoms)  // Kelvin
}

// Expected ΔT per UV burst:
// TRP: ~15-20 K local heating
// TYR: ~8-12 K local heating
// PHE: ~2-5 K local heating
```

### 3. Frequency Hopping Protocol

```rust
pub struct FrequencyHoppingProtocol {
    /// Wavelengths to scan (nm)
    pub wavelengths: Vec<f32>,  // [258, 265, 274, 280, 290]

    /// Dwell time per wavelength (steps)
    pub dwell_steps: i32,

    /// Number of full scans
    pub n_scans: i32,

    /// Recording buffer per wavelength
    pub response_buffer: HashMap<u32, Vec<SpikeResponse>>,
}

impl FrequencyHoppingProtocol {
    pub fn spectral_scan() -> Self {
        Self {
            wavelengths: vec![258.0, 265.0, 274.0, 280.0, 290.0],
            dwell_steps: 1000,  // 2 ps per wavelength
            n_scans: 5,
            response_buffer: HashMap::new(),
        }
    }

    /// Get current wavelength based on step
    pub fn current_wavelength(&self, step: i32) -> f32 {
        let scan_length = self.wavelengths.len() as i32 * self.dwell_steps;
        let position = (step % scan_length) / self.dwell_steps;
        self.wavelengths[position as usize]
    }

    /// Record spike response at current wavelength
    pub fn record_response(&mut self, wavelength: f32, spikes: &[SpikeEvent]) {
        let key = (wavelength * 10.0) as u32;  // 0.1 nm resolution
        self.response_buffer
            .entry(key)
            .or_default()
            .extend(spikes.iter().cloned());
    }
}
```

### 4. Disulfide Bond Targeting

```rust
/// Detect and target disulfide bonds
pub fn detect_disulfides(
    atom_names: &[String],
    residue_names: &[String],
    bonds: &[(usize, usize)],
    positions: &[f32],
) -> Vec<DisulfideTarget> {
    let mut disulfides = Vec::new();

    // Find CYS residues with SG atoms
    let sg_atoms: Vec<(usize, usize)> = atom_names.iter()
        .enumerate()
        .filter(|(_, name)| *name == "SG")
        .map(|(i, _)| (i, residue_ids[i]))
        .collect();

    // Check for S-S bonds (distance < 2.5 Å)
    for i in 0..sg_atoms.len() {
        for j in (i+1)..sg_atoms.len() {
            let (idx1, _) = sg_atoms[i];
            let (idx2, _) = sg_atoms[j];

            let dist = compute_distance(positions, idx1, idx2);
            if dist < 2.5 {
                disulfides.push(DisulfideTarget {
                    atom1: idx1,
                    atom2: idx2,
                    bond_length: dist,
                    lambda_max: 250.0,
                    extinction: 300.0,
                });
            }
        }
    }

    disulfides
}
```

### 5. Enhanced Data Recording

```rust
/// Full UV spectroscopy output
#[derive(Serialize, Deserialize)]
pub struct UvSpectroscopyResults {
    // Per-wavelength response
    pub spectral_response: HashMap<String, WavelengthResponse>,

    // Per-chromophore tracking
    pub chromophore_responses: Vec<ChromophoreResponse>,

    // Temperature fluctuations
    pub local_temperature_history: Vec<LocalTempRecord>,

    // Disulfide dynamics
    pub disulfide_events: Vec<DisulfideEvent>,

    // Spike-wavelength correlation
    pub wavelength_spike_correlation: Vec<(f32, f32)>,  // (λ, spike_count)
}

#[derive(Serialize, Deserialize)]
pub struct ChromophoreResponse {
    pub residue_name: String,
    pub residue_id: i32,
    pub chromophore_type: String,  // "aromatic" or "disulfide"
    pub lambda_max: f32,

    // Time series
    pub local_temp_delta: Vec<f32>,    // ΔT per frame
    pub displacement: Vec<f32>,         // Ring RMSD per frame
    pub nearby_water_density: Vec<f32>, // Water density within 5Å
    pub spike_triggered: Vec<bool>,     // Did this trigger a spike?
}

#[derive(Serialize, Deserialize)]
pub struct LocalTempRecord {
    pub frame: i32,
    pub residue_id: i32,
    pub delta_t: f32,           // Local temperature increase (K)
    pub dissipation_tau: f32,   // Decay time constant (ps)
    pub neighboring_residues_affected: Vec<i32>,
}
```

### 6. Integration with Neuromorphic Layer

```rust
/// Enhanced spike detection with wavelength correlation
pub struct WavelengthAwareSpike {
    pub base_spike: SpikeEvent,

    // UV correlation
    pub triggering_wavelength: Option<f32>,
    pub time_since_uv_burst: i32,  // frames
    pub chromophore_distance: f32,  // Å to nearest UV target

    // Thermal correlation
    pub local_temp_at_spike: f32,
    pub temp_gradient: [f32; 3],  // Direction of heat flow

    // Classification
    pub spike_category: SpikeCategory,
}

pub enum SpikeCategory {
    DirectUvInduced,      // <5Å from chromophore, <100 steps after burst
    IndirectThermal,      // Temperature-driven, not near chromophore
    CooperativeNetwork,   // Part of spike avalanche
    SpontaneousFluctuation, // Background noise
}
```

## Implementation Priority

### Phase 1: Minimal for Publication (This Week)
- [x] Existing 280nm model (working)
- [ ] Add local temperature tracking per burst
- [ ] Record wavelength-spike correlation
- [ ] Classify spikes by proximity to aromatics

### Phase 2: Full Spectroscopy (Post-Publication)
- [ ] Multi-wavelength frequency hopping
- [ ] Disulfide bond targeting
- [ ] Electronic state modeling
- [ ] Full chromophore response tracking

## Output Enhancement for Publication

Current output includes:
- total_spikes
- aromatic_adjacent_spikes
- weighted_score

Enhanced output should include:
- per_chromophore_response (TRP vs TYR vs PHE breakdown)
- local_temperature_delta (heat dissipation tracking)
- wavelength_spike_histogram (if frequency hopping)
- disulfide_events (bond weakening observations)
- spike_classification_counts (direct vs indirect vs cooperative)
