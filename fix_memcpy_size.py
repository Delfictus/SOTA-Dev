#!/usr/bin/env python3
"""Fix memcpy_dtoh assertion: host buffer must be >= device buffer size."""

FILE = "crates/prism-nhs/src/fused_engine.rs"

with open(FILE, 'r') as f:
    content = f.read()

old = (
    '                let n_to_download = spikes.min(MAX_SPIKES_PER_STEP);\n'
    '                let bytes_needed = n_to_download * self.spike_event_size;\n'
    '\n'
    '                // Download displacement-based spike events (from nhs_spike_detect kernel)\n'
    '                let mut full_buffer = vec![0u8; bytes_needed];\n'
    '                self.stream.memcpy_dtoh(&self.d_spike_detect_events, &mut full_buffer)?;'
)

new = (
    '                let n_to_download = spikes.min(MAX_SPIKES_PER_STEP);\n'
    '\n'
    '                // Download displacement-based spike events (from nhs_spike_detect kernel)\n'
    '                // cudarc requires host buffer >= device buffer, so allocate full size\n'
    '                let full_device_size = MAX_SPIKES_PER_STEP * self.spike_event_size;\n'
    '                let mut full_buffer = vec![0u8; full_device_size];\n'
    '                self.stream.memcpy_dtoh(&self.d_spike_detect_events, &mut full_buffer)?;'
)

count = content.count(old)
if count == 1:
    content = content.replace(old, new)
    with open(FILE, 'w') as f:
        f.write(content)
    print("✓ Fixed memcpy_dtoh buffer size mismatch")
    print("Next: cargo build --release -p prism-nhs 2>&1 | tail -10")
else:
    print(f"✗ Pattern not found ({count} matches)")
