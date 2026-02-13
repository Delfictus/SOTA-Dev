#!/usr/bin/env python3
"""Fix 2 compilation errors in the spike_detect integration."""

import sys

FILE = "crates/prism-nhs/src/fused_engine.rs"

with open(FILE, 'r') as f:
    content = f.read()

fixes = 0

# FIX 1: Replace topology.aromatic_targets loop with atom_to_aromatic vec
# By this point in the constructor, atom_to_aromatic: Vec<i32> is already built
# (line ~1755). Entry is aromatic_idx if aromatic, -1 otherwise.
old1 = (
    '        // Build flat list of all individual aromatic ATOM indices from topology\n'
    '        let mut spike_detect_atom_indices: Vec<i32> = Vec::new();\n'
    '        for target in &topology.aromatic_targets {\n'
    '            for &atom_idx in &target.ring_atom_indices {\n'
    '                if atom_idx >= 0 && (atom_idx as usize) < n_atoms {\n'
    '                    spike_detect_atom_indices.push(atom_idx);\n'
    '                }\n'
    '            }\n'
    '        }'
)
new1 = (
    '        // Build flat list of all individual aromatic ATOM indices\n'
    '        // Uses atom_to_aromatic mapping already built above\n'
    '        let spike_detect_atom_indices: Vec<i32> = atom_to_aromatic.iter()\n'
    '            .enumerate()\n'
    '            .filter(|(_, &arom_idx)| arom_idx >= 0)\n'
    '            .map(|(atom_idx, _)| atom_idx as i32)\n'
    '            .collect();'
)

if content.count(old1) == 1:
    content = content.replace(old1, new1)
    print("  ✓ FIX 1: replaced topology.aromatic_targets with atom_to_aromatic")
    fixes += 1
else:
    print(f"  ✗ FIX 1: pattern not found ({content.count(old1)} matches)")

# FIX 2: Replace spike_event_size with inline size_of (not yet defined as local)
old2 = '        let d_spike_detect_events: CudaSlice<u8> = stream.alloc_zeros(MAX_SPIKES_PER_STEP * spike_event_size)?;'
new2 = '        let d_spike_detect_events: CudaSlice<u8> = stream.alloc_zeros(MAX_SPIKES_PER_STEP * std::mem::size_of::<GpuSpikeEvent>())?;'

if content.count(old2) == 1:
    content = content.replace(old2, new2)
    print("  ✓ FIX 2: replaced spike_event_size with std::mem::size_of::<GpuSpikeEvent>()")
    fixes += 1
else:
    print(f"  ✗ FIX 2: pattern not found ({content.count(old2)} matches)")

with open(FILE, 'w') as f:
    f.write(content)

print(f"\n{fixes}/2 fixes applied.")
if fixes == 2:
    print("Next: cargo build --release -p prism-nhs 2>&1 | tail -30")
