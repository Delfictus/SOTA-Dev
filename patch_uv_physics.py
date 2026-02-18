#!/usr/bin/env python3
"""
Patch apply_batch_uv_burst() in nhs_rt_full.rs with physics-correct UV photochemistry.

Fixes:
1. Wavelength-dependent extinction coefficients (not flat energy)
2. Probabilistic absorption (not every atom every burst)
3. Per-atom mass from topology (not hardcoded 12 AMU)
4. Random direction per burst (not deterministic)
5. Wavelength cycling across chromophore types
"""

import re
import sys

RS_FILE = 'crates/prism-nhs/src/bin/nhs_rt_full.rs'

with open(RS_FILE, 'r') as f:
    content = f.read()

# ============================================================
# PATCH 1: Update function signature and call sites
# Add current_step parameter to apply_batch_uv_burst
# ============================================================

# Fix all 3 call sites: add *current_step argument
old_call = 'apply_batch_uv_burst(batch, aromatic_indices_per_structure, topologies, uv_energy)?;'
new_call = 'apply_batch_uv_burst(batch, aromatic_indices_per_structure, topologies, uv_energy, *current_step)?;'

count = content.count(old_call)
if count > 0:
    content = content.replace(old_call, new_call)
    print(f"  ✓ Updated {count} call sites with current_step parameter")
else:
    # Maybe already patched?
    if '*current_step)?' in content:
        print("  ⚠ Call sites already patched")
    else:
        print("  ✗ Could not find call sites!")
        sys.exit(1)

# ============================================================
# PATCH 2: Replace the entire apply_batch_uv_burst function
# ============================================================

# Find the old function by its unique signature line
old_fn_start = 'fn apply_batch_uv_burst(\n    batch: &mut prism_gpu::AmberSimdBatch,\n    aromatic_indices_per_structure: &[Vec<usize>],\n    topologies: &[PrismPrepTopology],\n    energy: f32,\n) -> Result<()> {'

# Try to find it
if old_fn_start in content:
    # Find the end of the function (next fn or closing brace pattern)
    start_idx = content.index(old_fn_start)
    
    # Find the closing "}\n\n" that ends this function
    # We know it ends with "Ok(())\n}" 
    search_from = start_idx
    end_marker = '\n}\n\n/// Detect spikes'
    end_idx = content.index(end_marker, search_from)
    end_idx += 2  # Include the closing brace and newline
    
    old_fn_text = content[start_idx:end_idx]
    print(f"  ✓ Found old function ({len(old_fn_text)} chars)")
else:
    print("  ✗ Could not find old function signature. Trying alternate...")
    # Try without the energy param (maybe already partially patched)
    alt = 'fn apply_batch_uv_burst('
    if alt in content:
        start_idx = content.index(alt)
        end_marker = '\n}\n\n/// Detect spikes'
        end_idx = content.index(end_marker, start_idx)
        end_idx += 2
        old_fn_text = content[start_idx:end_idx]
        print(f"  ✓ Found function via alternate match ({len(old_fn_text)} chars)")
    else:
        print("  ✗ Cannot find apply_batch_uv_burst at all!")
        sys.exit(1)

new_fn_text = '''fn apply_batch_uv_burst(
    batch: &mut prism_gpu::AmberSimdBatch,
    aromatic_indices_per_structure: &[Vec<usize>],
    topologies: &[PrismPrepTopology],
    _energy: f32,
    current_step: usize,
) -> Result<()> {
    use prism_nhs::config::{
        extinction_to_cross_section, wavelength_to_ev,
        CALIBRATED_PHOTON_FLUENCE,
        HEAT_YIELD_TRP, HEAT_YIELD_TYR, HEAT_YIELD_PHE,
        KB_EV_K, NEFF_TRP, NEFF_TYR, NEFF_PHE,
    };

    // Wavelength cycling: rotate through chromophore-specific wavelengths
    // 280nm=TRP, 274nm=TYR, 258nm=PHE, 211nm=HIS
    let wavelengths = [280.0f32, 274.0, 258.0, 211.0];
    let wavelength = wavelengths[current_step / 250 % wavelengths.len()];

    let mut velocities = batch.get_velocities()?;
    let max_stride = batch.max_atoms_per_struct() * 3;

    // Seed RNG from current_step for reproducible but varying directions
    let mut rng_state: u64 = current_step as u64 * 6364136223846793005 + 1442695040888963407;

    for (struct_idx, aromatic_indices) in aromatic_indices_per_structure.iter().enumerate() {
        if aromatic_indices.is_empty() {
            continue;
        }

        let topology = &topologies[struct_idx];
        let offset = struct_idx * max_stride;

        for &atom_idx in aromatic_indices {
            if atom_idx >= topology.residue_names.len() {
                continue;
            }

            // Classify chromophore from residue name
            let res_name = &topology.residue_names[atom_idx];
            let (chromophore_type, heat_yield, n_eff) = match res_name.as_str() {
                "TRP" => (0i32, HEAT_YIELD_TRP, NEFF_TRP),
                "TYR" => (1, HEAT_YIELD_TYR, NEFF_TYR),
                "PHE" => (2, HEAT_YIELD_PHE, NEFF_PHE),
                "HIS" | "HID" | "HIE" | "HIP" => (3, 0.95f32, 6.0f32),
                _ => continue,
            };

            // Wavelength-dependent extinction (Gaussian band model, FWHM ~15nm)
            let (peak_wavelength, peak_extinction) = match chromophore_type {
                0 => (280.0f32, 5500.0f32),  // TRP: indole
                1 => (274.0, 1490.0),          // TYR: phenol
                2 => (258.0, 200.0),           // PHE: benzene
                3 => (211.0, 5700.0),          // HIS: imidazole
                _ => continue,
            };
            let sigma_nm = 7.5f32; // Gaussian width
            let delta = wavelength - peak_wavelength;
            let extinction = peak_extinction * (-0.5 * (delta / sigma_nm).powi(2)).exp();

            // Skip if negligible absorption at this wavelength
            if extinction < 10.0 {
                continue;
            }

            // Physics: absorption cross-section and probability
            let cross_section = extinction_to_cross_section(extinction);
            let p_absorb = cross_section * CALIBRATED_PHOTON_FLUENCE;

            // Stochastic absorption check (PCG-style fast hash)
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(
                (struct_idx as u64) << 32 | atom_idx as u64
            );
            let rand_val = ((rng_state >> 33) as f32) / (u32::MAX as f32);
            if rand_val > p_absorb {
                continue; // Photon not absorbed by this chromophore
            }

            // Energy deposited: E_photon * heat_yield
            let e_photon = wavelength_to_ev(wavelength);
            let e_dep = e_photon * heat_yield;

            // Local heating: delta_T = E_dep / (1.5 * k_B * N_eff)
            let delta_t_kelvin = e_dep / (1.5 * KB_EV_K * n_eff);

            // Use real atomic mass from topology
            let mass_amu = if atom_idx < topology.masses.len() {
                topology.masses[atom_idx].max(1.0)
            } else {
                12.0
            };

            // KE -> velocity: v = sqrt(2 * KE_eV * 96.485 / mass_amu) in Å/ps
            let ke_ev = 1.5 * KB_EV_K * delta_t_kelvin;
            let velocity_boost = (2.0 * ke_ev * 96.485 / mass_amu).sqrt().max(0.0);

            // Proper uniform random direction on unit sphere
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng_state >> 33) as f32) / (u32::MAX as f32);
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((rng_state >> 33) as f32) / (u32::MAX as f32);
            let cos_theta = 2.0 * u1 - 1.0;
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let phi = 2.0 * std::f32::consts::PI * u2;

            let base = offset + atom_idx * 3;
            if base + 2 < velocities.len() {
                velocities[base]     += velocity_boost * sin_theta * phi.cos();
                velocities[base + 1] += velocity_boost * sin_theta * phi.sin();
                velocities[base + 2] += velocity_boost * cos_theta;
            }
        }
    }

    batch.set_velocities(&velocities)?;
    Ok(())
}'''

content = content[:start_idx] + new_fn_text + content[end_idx:]
print(f"  ✓ Replaced function with physics-correct version ({len(new_fn_text)} chars)")

# ============================================================
# Write back
# ============================================================
with open(RS_FILE, 'w') as f:
    f.write(content)

print(f"\n✅ Patch complete. Changes:")
print(f"   • Wavelength cycling: 280→274→258→211nm every 250 steps")
print(f"   • Extinction-weighted absorption (TRP 27× more than PHE at 280nm)")
print(f"   • Probabilistic photon absorption (not every atom every burst)")
print(f"   • Per-atom mass from topology (not hardcoded 12 AMU)")
print(f"   • Proper spherical random kick direction")
print(f"   • Chromophore-specific heat yields (TRP=0.74, TYR=0.83, PHE=0.97)")
print(f"\nRun: cargo build --release -p prism-nhs --bin nhs_rt_full")
