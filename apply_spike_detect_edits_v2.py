#!/usr/bin/env python3
"""
Apply nhs_spike_detect integration edits to fused_engine.rs (v2 - fixed patterns)

Usage: python3 apply_spike_detect_edits_v2.py
Run from: ~/Desktop/Prism4D-bio/
"""

import os
import sys
import shutil

FILE = "crates/prism-nhs/src/fused_engine.rs"

def apply_edit(content, old, new, label):
    count = content.count(old)
    if count == 0:
        print(f"  ✗ {label}: pattern NOT FOUND")
        print(f"    First 80 chars: {repr(old[:80])}")
        return content, False
    if count > 1:
        print(f"  ⚠ {label}: found {count}x, replacing first only")
        content = content.replace(old, new, 1)
        return content, True
    content = content.replace(old, new, 1)
    print(f"  ✓ {label}")
    return content, True

def apply_edit_all(content, old, new, label):
    count = content.count(old)
    if count == 0:
        print(f"  ✗ {label}: pattern NOT FOUND")
        return content, False
    content = content.replace(old, new)
    print(f"  ✓ {label} (replaced {count} occurrence(s))")
    return content, True

def main():
    if not os.path.exists(FILE):
        print(f"ERROR: {FILE} not found. Run from Prism4D-bio root.")
        sys.exit(1)

    backup = FILE + ".bak_lif"
    if not os.path.exists(backup):
        shutil.copy2(FILE, backup)
        print(f"Created backup: {backup}")
    else:
        print(f"Backup exists: {backup}")

    with open(FILE, 'r') as f:
        content = f.read()

    original_len = len(content)
    all_ok = True

    print("\nApplying edits...\n")

    # ================================================================
    # EDIT 1: Add struct fields
    # Note: blank line between d_aromatic_n_atoms and the O(N) section
    # ================================================================
    old = (
        '    d_aromatic_n_atoms: CudaSlice<i32>,       // [n_aromatics] - count of atoms per aromatic\n'
        '\n'
        '    // ====================================================================\n'
        '    // O(N) CELL LIST / NEIGHBOR LIST BUFFERS\n'
        '    // ===================================================================='
    )
    new = (
        '    d_aromatic_n_atoms: CudaSlice<i32>,       // [n_aromatics] - count of atoms per aromatic\n'
        '\n'
        '    // ====================================================================\n'
        '    // DISPLACEMENT-BASED SPIKE DETECTION (nhs_spike_detect.cu)\n'
        '    // ====================================================================\n'
        '    spike_detect_kernel: CudaFunction,\n'
        '    spike_reset_kernel: CudaFunction,\n'
        '    d_prev_positions: CudaSlice<f32>,          // [n_atoms * 3] - previous step positions\n'
        '    d_spike_detect_aromatic_indices: CudaSlice<i32>,  // [n_spike_detect_aromatics] - flat aromatic atom indices\n'
        '    d_spike_detect_events: CudaSlice<u8>,      // [MAX_SPIKES_PER_STEP * spike_event_size]\n'
        '    d_spike_detect_count: CudaSlice<i32>,      // [1] - atomic counter\n'
        '    n_spike_detect_aromatics: usize,           // Number of individual aromatic atoms\n'
        '\n'
        '    // ====================================================================\n'
        '    // O(N) CELL LIST / NEIGHBOR LIST BUFFERS\n'
        '    // ===================================================================='
    )
    content, ok = apply_edit(content, old, new, "EDIT 1: struct fields")
    all_ok = all_ok and ok

    # ================================================================
    # EDIT 2: Allocate buffers + load PTX in constructor
    # FIXED: use ring_atom_indices (not atom_indices) from topology
    # ================================================================
    old = (
        '        // These are needed by build_aromatic_neighbors and compute_ring_normals CUDA kernels\n'
        '        let d_aromatic_atom_indices: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1) * 16)?;\n'
        '        let d_aromatic_n_atoms: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1))?;'
    )
    new = (
        '        // These are needed by build_aromatic_neighbors and compute_ring_normals CUDA kernels\n'
        '        let d_aromatic_atom_indices: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1) * 16)?;\n'
        '        let d_aromatic_n_atoms: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1))?;\n'
        '\n'
        '        // ================================================================\n'
        '        // DISPLACEMENT-BASED SPIKE DETECTION: Load kernel + allocate buffers\n'
        '        // ================================================================\n'
        '        // Build flat list of all individual aromatic ATOM indices from topology\n'
        '        let mut spike_detect_atom_indices: Vec<i32> = Vec::new();\n'
        '        for target in &topology.aromatic_targets {\n'
        '            for &atom_idx in &target.ring_atom_indices {\n'
        '                if atom_idx >= 0 && (atom_idx as usize) < n_atoms {\n'
        '                    spike_detect_atom_indices.push(atom_idx);\n'
        '                }\n'
        '            }\n'
        '        }\n'
        '        let n_spike_detect_aromatics = spike_detect_atom_indices.len();\n'
        '        log::info!("Spike detect: {} individual aromatic atoms from {} groups",\n'
        '                   n_spike_detect_aromatics, n_aromatics);\n'
        '\n'
        '        // Allocate GPU buffers for spike detection\n'
        '        let d_prev_positions: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;\n'
        '        let mut d_spike_detect_aromatic_indices: CudaSlice<i32> = stream.alloc_zeros(n_spike_detect_aromatics.max(1))?;\n'
        '        if n_spike_detect_aromatics > 0 {\n'
        '            stream.memcpy_htod(&spike_detect_atom_indices, &mut d_spike_detect_aromatic_indices)?;\n'
        '        }\n'
        '        let d_spike_detect_events: CudaSlice<u8> = stream.alloc_zeros(MAX_SPIKES_PER_STEP * spike_event_size)?;\n'
        '        let d_spike_detect_count: CudaSlice<i32> = stream.alloc_zeros(1)?;\n'
        '\n'
        '        // Load nhs_spike_detect PTX\n'
        '        let spike_detect_module = {\n'
        '            let spike_ptx_paths = vec![\n'
        '                std::path::PathBuf::from("crates/prism-gpu/src/kernels/nhs_spike_detect_sm120.ptx"),\n'
        '                std::path::PathBuf::from("crates/prism-gpu/src/kernels/nhs_spike_detect.ptx"),\n'
        '            ];\n'
        '            let mut loaded = None;\n'
        '            for path in &spike_ptx_paths {\n'
        '                if path.exists() {\n'
        '                    match context.load_module(Ptx::from_file(&path.display().to_string())) {\n'
        '                        Ok(m) => {\n'
        '                            log::info!("Loaded spike_detect PTX from: {}", path.display());\n'
        '                            loaded = Some(m);\n'
        '                            break;\n'
        '                        }\n'
        '                        Err(e) => log::warn!("Failed to load spike_detect PTX from {}: {}", path.display(), e),\n'
        '                    }\n'
        '                }\n'
        '            }\n'
        '            loaded.ok_or_else(|| anyhow::anyhow!("Failed to load nhs_spike_detect PTX"))?\n'
        '        };\n'
        '        let spike_detect_kernel = spike_detect_module.load_function("nhs_spike_detect")?;\n'
        '        let spike_reset_kernel = spike_detect_module.load_function("nhs_spike_reset_counter")?;'
    )
    content, ok = apply_edit(content, old, new, "EDIT 2: alloc + load PTX")
    all_ok = all_ok and ok

    # ================================================================
    # EDIT 3: Add fields to struct literal initialization
    # ================================================================
    old = (
        '            d_aromatic_atom_indices,\n'
        '            d_aromatic_n_atoms,'
    )
    new = (
        '            d_aromatic_atom_indices,\n'
        '            d_aromatic_n_atoms,\n'
        '            // Displacement-based spike detection\n'
        '            spike_detect_kernel,\n'
        '            spike_reset_kernel,\n'
        '            d_prev_positions,\n'
        '            d_spike_detect_aromatic_indices,\n'
        '            d_spike_detect_events,\n'
        '            d_spike_detect_count,\n'
        '            n_spike_detect_aromatics,'
    )
    content, ok = apply_edit(content, old, new, "EDIT 3: struct literal")
    all_ok = all_ok and ok

    # ================================================================
    # EDIT 4: Launch spike_detect after fused step
    # Note: blank line after .context(...) before // Advance
    # ================================================================
    old = (
        '        .context("Failed to launch nhs_amber_fused_step kernel")?;\n'
        '\n'
        '        // Advance protocols (CPU-side, no sync needed)\n'
        '        self.temp_protocol.advance();\n'
        '        self.uv_config.advance();\n'
        '        self.timestep += 1;'
    )
    new = (
        '        .context("Failed to launch nhs_amber_fused_step kernel")?;\n'
        '\n'
        '        // ================================================================\n'
        '        // DISPLACEMENT-BASED SPIKE DETECTION (runs every step on GPU)\n'
        '        // ================================================================\n'
        '        if self.n_spike_detect_aromatics > 0 {\n'
        '            let n_aromatics_detect_i32 = self.n_spike_detect_aromatics as i32;\n'
        '            let n_atoms_i32 = self.n_atoms as i32;\n'
        '            let displacement_threshold: f32 = 0.5;  // Angstroms\n'
        '            let proximity_threshold: f32 = 6.0;     // Angstroms\n'
        '            let max_nearby: i32 = 20;\n'
        '            let max_spikes_detect_i32 = MAX_SPIKES_PER_STEP as i32;\n'
        '\n'
        '            unsafe {\n'
        '                self.stream\n'
        '                    .launch_builder(&self.spike_detect_kernel)\n'
        '                    .arg(&self.d_positions)\n'
        '                    .arg(&self.d_prev_positions)\n'
        '                    .arg(&self.d_spike_detect_aromatic_indices)\n'
        '                    .arg(&n_aromatics_detect_i32)\n'
        '                    .arg(&n_atoms_i32)\n'
        '                    .arg(&self.timestep)\n'
        '                    .arg(&displacement_threshold)\n'
        '                    .arg(&proximity_threshold)\n'
        '                    .arg(&max_nearby)\n'
        '                    .arg(&mut self.d_spike_detect_events)\n'
        '                    .arg(&mut self.d_spike_detect_count)\n'
        '                    .arg(&max_spikes_detect_i32)\n'
        '                    .launch(LaunchConfig {\n'
        '                        grid_dim: (((self.n_spike_detect_aromatics + 255) / 256) as u32, 1, 1),\n'
        '                        block_dim: (256, 1, 1),\n'
        '                        shared_mem_bytes: 0,\n'
        '                    })\n'
        '            }\n'
        '            .context("Failed to launch nhs_spike_detect kernel")?;\n'
        '\n'
        '            // Copy current positions -> prev for next step displacement calc\n'
        '            self.stream.memcpy_dtod(&self.d_positions, &mut self.d_prev_positions)?;\n'
        '        }\n'
        '\n'
        '        // Advance protocols (CPU-side, no sync needed)\n'
        '        self.temp_protocol.advance();\n'
        '        self.uv_config.advance();\n'
        '        self.timestep += 1;'
    )
    # This pattern may appear twice (single-step + replica paths). Replace first only.
    count = content.count(old)
    if count == 0:
        print("  ✗ EDIT 4: launch spike_detect: pattern NOT FOUND")
        all_ok = False
    else:
        content = content.replace(old, new, 1)
        print(f"  ✓ EDIT 4: launch spike_detect after step ({count} match(es), replaced first)")

    # ================================================================
    # EDIT 5: Read spike_detect count instead of fused kernel count
    # ================================================================
    old = (
        '            let mut spike_count_host = [0i32];\n'
        '            self.stream.memcpy_dtoh(&self.d_spike_count, &mut spike_count_host)?;\n'
        '            let spikes = spike_count_host[0] as usize;'
    )
    new = (
        '            // Read displacement-based spike count (from nhs_spike_detect kernel)\n'
        '            let mut spike_count_host = [0i32];\n'
        '            self.stream.memcpy_dtoh(&self.d_spike_detect_count, &mut spike_count_host)?;\n'
        '            let spikes = spike_count_host[0] as usize;'
    )
    content, ok = apply_edit_all(content, old, new, "EDIT 5: spike count read")
    all_ok = all_ok and ok

    # ================================================================
    # EDIT 5b: Reset both spike counters
    # ================================================================
    old = (
        '            // Reset spike count for next sync interval\n'
        '            // This must happen AFTER we\'ve read the spike count to preserve accumulated spikes\n'
        '            let zero = [0i32];\n'
        '            self.stream.memcpy_htod(&zero, &mut self.d_spike_count)?;'
    )
    new = (
        '            // Reset BOTH spike counters for next sync interval\n'
        '            // This must happen AFTER we\'ve read the spike count to preserve accumulated spikes\n'
        '            let zero = [0i32];\n'
        '            self.stream.memcpy_htod(&zero, &mut self.d_spike_count)?;         // fused LIF (still runs, ignored)\n'
        '            self.stream.memcpy_htod(&zero, &mut self.d_spike_detect_count)?;  // displacement-based (primary)'
    )
    content, ok = apply_edit(content, old, new, "EDIT 5b: reset both counters")
    all_ok = all_ok and ok

    # ================================================================
    # EDIT 5c: Read from spike_detect_events for accumulation
    # ================================================================
    old = (
        '            if self.accumulate_spikes && spikes > 0 {\n'
        '                let n_to_download = spikes.min(MAX_SPIKES_PER_STEP);\n'
        '                let bytes_needed = n_to_download * self.spike_event_size;\n'
        '\n'
        '                // Download ONLY the actual spike bytes (not the full 6MB buffer!)\n'
        '                // This is a major performance optimization - copies bytes_needed instead of 6MB\n'
        '                let mut full_buffer = vec![0u8; bytes_needed];\n'
        '                self.stream.memcpy_dtoh(&self.d_spike_events, &mut full_buffer)?;'
    )
    new = (
        '            if self.accumulate_spikes && spikes > 0 {\n'
        '                let n_to_download = spikes.min(MAX_SPIKES_PER_STEP);\n'
        '                let bytes_needed = n_to_download * self.spike_event_size;\n'
        '\n'
        '                // Download displacement-based spike events (from nhs_spike_detect kernel)\n'
        '                let mut full_buffer = vec![0u8; bytes_needed];\n'
        '                self.stream.memcpy_dtoh(&self.d_spike_detect_events, &mut full_buffer)?;'
    )
    content, ok = apply_edit(content, old, new, "EDIT 5c: read spike_detect_events")
    all_ok = all_ok and ok

    # ================================================================
    # EDIT 6: Initialize prev_positions when topology uploaded
    # ================================================================
    old = '        self.stream.memcpy_htod(&topology.positions, &mut self.d_positions)?;'
    new = (
        '        self.stream.memcpy_htod(&topology.positions, &mut self.d_positions)?;\n'
        '        // Initialize prev_positions to same as initial (no displacement on first step)\n'
        '        self.stream.memcpy_dtod(&self.d_positions, &mut self.d_prev_positions)?;'
    )
    content, ok = apply_edit(content, old, new, "EDIT 6: init prev_positions")
    all_ok = all_ok and ok

    # ================================================================
    # WRITE
    # ================================================================
    with open(FILE, 'w') as f:
        f.write(content)

    new_len = len(content)
    status = "✓ ALL EDITS APPLIED" if all_ok else "⚠ SOME EDITS FAILED"
    print(f"\n{status}")
    print(f"{original_len} -> {new_len} bytes (+{new_len - original_len})")
    print(f"Backup: {backup}")

    if all_ok:
        print(f"\nNext: cargo build --release -p prism-nhs 2>&1 | tail -30")

if __name__ == "__main__":
    main()
