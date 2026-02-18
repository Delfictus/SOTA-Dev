#!/usr/bin/env python3
"""Fix seed propagation: seeded MB velocities in reset_for_replica (line-based)"""

path = "crates/prism-nhs/src/fused_engine.rs"
with open(path, "r") as f:
    lines = f.readlines()

# Find 'pub fn reset_for_replica'
start = None
for i, line in enumerate(lines):
    if 'pub fn reset_for_replica(&mut self, seed: u64)' in line:
        start = i
        break

if start is None:
    print("ERROR: Cannot find reset_for_replica")
    raise SystemExit(1)

# Find the closing '    }' of the function (matching indent level)
end = None
brace_depth = 0
for i in range(start, min(start + 60, len(lines))):
    brace_depth += lines[i].count('{') - lines[i].count('}')
    if brace_depth == 0 and i > start:
        end = i
        break

if end is None:
    print("ERROR: Cannot find end of reset_for_replica")
    raise SystemExit(1)

print(f"Found reset_for_replica at lines {start+1}-{end+1}")

# Build replacement function body
new_body = [
    '    pub fn reset_for_replica(&mut self, seed: u64) -> Result<()> {\n',
    '        // Clear accumulated data\n',
    '        self.accumulated_spikes.clear();\n',
    '        self.ensemble_snapshots.clear();\n',
    '\n',
    '        // Re-initialize GPU RNG with new seed (affects Langevin noise in kernels)\n',
    '        self.init_rng(seed)?;\n',
    '\n',
    '        // Reset simulation counters\n',
    '        self.timestep = 0;\n',
    '        self.last_spike_count = 0;\n',
    '\n',
    '        // Generate seeded Maxwell-Boltzmann velocities for true trajectory divergence.\n',
    '        // Each replica gets deterministically different initial velocities from its seed,\n',
    '        // ensuring warm-phase dynamics diverge even with identical starting positions.\n',
    '        {\n',
    '            use rand::SeedableRng;\n',
    '            use rand_distr::{Distribution, Normal};\n',
    '            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);\n',
    '            let mut velocities = vec![0.0f32; self.n_atoms * 3];\n',
    '            const KB: f32 = 0.001987204;\n',
    '            let temp = self.temp_protocol.start_temp.max(50.0);\n',
    '            // Download masses from GPU\n',
    '            let mut masses = vec![0.0f32; self.n_atoms];\n',
    '            self.stream.memcpy_dtoh(&self.d_masses, &mut masses)?;\n',
    '            for i in 0..self.n_atoms {\n',
    '                let mass = masses[i];\n',
    '                if mass <= 0.0 { continue; }\n',
    '                let sigma = (KB * temp / mass).sqrt();\n',
    '                let normal = Normal::new(0.0f64, sigma as f64).unwrap();\n',
    '                velocities[i * 3] = normal.sample(&mut rng) as f32;\n',
    '                velocities[i * 3 + 1] = normal.sample(&mut rng) as f32;\n',
    '                velocities[i * 3 + 2] = normal.sample(&mut rng) as f32;\n',
    '            }\n',
    '            // Remove center of mass velocity\n',
    '            let mut com_vel = [0.0f32; 3];\n',
    '            let mut total_mass = 0.0f32;\n',
    '            for i in 0..self.n_atoms {\n',
    '                let m = masses[i];\n',
    '                com_vel[0] += m * velocities[i * 3];\n',
    '                com_vel[1] += m * velocities[i * 3 + 1];\n',
    '                com_vel[2] += m * velocities[i * 3 + 2];\n',
    '                total_mass += m;\n',
    '            }\n',
    '            if total_mass > 0.0 {\n',
    '                for i in 0..self.n_atoms {\n',
    '                    velocities[i * 3] -= com_vel[0] / total_mass;\n',
    '                    velocities[i * 3 + 1] -= com_vel[1] / total_mass;\n',
    '                    velocities[i * 3 + 2] -= com_vel[2] / total_mass;\n',
    '                }\n',
    '            }\n',
    '            self.stream.memcpy_htod(&velocities, &mut self.d_velocities)?;\n',
    '        }\n',
    '\n',
    '        // Reset temperature protocol\n',
    '        self.temp_protocol.current_step = 0;\n',
    '\n',
    '        log::debug!("Reset for replica with seed {} (seeded MB velocities + GPU RNG)", seed);\n',
    '        Ok(())\n',
    '    }\n',
]

# Replace lines[start : end+1] with new_body
new_lines = lines[:start] + new_body + lines[end+1:]

with open(path, "w") as f:
    f.writelines(new_lines)

print(f"✓ Fixed: {path}")
print("  - Seeded MB velocities from StdRng::seed_from_u64(seed)")
print("  - GPU RNG re-seeded for Langevin noise")
print("  - Each stream diverges from step 1")
