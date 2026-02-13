#!/usr/bin/env python3
"""Fix PATCH 1: Revert broken splice, re-apply correctly."""

path = "crates/prism-nhs/src/fused_engine.rs"
with open(path, "r") as f:
    lines = f.readlines()

# Step 1: Check if already (broken) patched — look for new_on_stream
has_new_on_stream = any('new_on_stream' in l for l in lines)

if has_new_on_stream:
    print("Detected previous (broken) patch — reverting first...")
    # Find and remove the new_on_stream block, restore original new()
    # Find 'pub fn new(' for NhsAmberFusedEngine
    new_start = None
    for i, line in enumerate(lines):
        if '    pub fn new(' in line:
            ctx = ''.join(lines[max(0,i-5):i])
            if 'NhsAmberFusedEngine' in ctx or 'PRISM-PREP topology' in ctx:
                new_start = i
                break
    
    if new_start is None:
        print("ERROR: Can't find NhsAmberFusedEngine::new()")
        raise SystemExit(1)
    
    # Find the Self::new_on_stream delegation line
    delegate_line = None
    for i in range(new_start, min(new_start + 15, len(lines))):
        if 'Self::new_on_stream' in lines[i]:
            delegate_line = i
            break
    
    # Find the closing } of delegating new()
    close_new = None
    if delegate_line:
        for i in range(delegate_line, min(delegate_line + 5, len(lines))):
            if lines[i].strip() == '}':
                close_new = i
                break
    
    # Find 'pub fn new_on_stream(' 
    nos_start = None
    for i in range(new_start, min(new_start + 30, len(lines))):
        if 'new_on_stream(' in lines[i] and 'pub fn' in lines[i]:
            nos_start = i
            break
    
    # Find ') -> Result<Self> {' of new_on_stream
    nos_body = None
    if nos_start:
        for i in range(nos_start, min(nos_start + 15, len(lines))):
            if ') -> Result<Self>' in lines[i]:
                nos_body = i
                break
    
    if all(x is not None for x in [new_start, delegate_line, close_new, nos_start, nos_body]):
        print(f"  Removing delegating new() lines {new_start+1}-{close_new+1}")
        print(f"  Removing new_on_stream header lines {nos_start+1}-{nos_body+1}")
        
        # Also find and remove comment lines before new_on_stream
        comment_start = nos_start
        for i in range(nos_start - 1, max(close_new, nos_start - 5), -1):
            if lines[i].strip().startswith('///') or lines[i].strip() == '':
                comment_start = i
            else:
                break
        
        # Rebuild: keep lines before new_start, 
        # then reconstruct original new() with the body from new_on_stream
        
        # Grab body lines from after new_on_stream's Result<Self> { 
        # up to wherever the original body continues
        body_after_nos = lines[nos_body + 1:]  # everything after ) -> Result<Self> {
        
        # Reconstruct original new()
        restored = lines[:new_start]
        restored.append('    pub fn new(\n')
        restored.append('        context: Arc<CudaContext>,\n')
        restored.append('        topology: &PrismPrepTopology,\n')
        restored.append('        grid_dim: usize,\n')
        restored.append('        grid_spacing: f32,\n')
        restored.append('    ) -> Result<Self> {\n')
        restored.extend(body_after_nos)
        
        lines = restored
        print("  Reverted to original new()")
    else:
        print(f"  WARNING: Could not fully parse broken patch state")
        print(f"    new_start={new_start}, delegate={delegate_line}, close={close_new}, nos_start={nos_start}, nos_body={nos_body}")
        print("  Attempting line-based reconstruction anyway...")

# Step 2: Now apply the patch correctly on clean code
# Re-scan for pub fn new(
target_line = None
for i, line in enumerate(lines):
    if '    pub fn new(' in line:
        ctx = ''.join(lines[max(0,i-5):i])
        if 'NhsAmberFusedEngine' in ctx or 'PRISM-PREP topology' in ctx:
            target_line = i
            break

if target_line is None:
    print("ERROR: Could not find NhsAmberFusedEngine::new()")
    raise SystemExit(1)

# Find ') -> Result<Self> {' 
result_line = None
for i in range(target_line, min(target_line + 10, len(lines))):
    if ') -> Result<Self>' in lines[i]:
        result_line = i
        break

if result_line is None:
    print("ERROR: Could not find ) -> Result<Self> {")
    raise SystemExit(1)

# Find 'let stream = context.default_stream();'
stream_line = None
for i in range(result_line, min(result_line + 20, len(lines))):
    if 'let stream = context.default_stream()' in lines[i]:
        stream_line = i
        break

if stream_line is None:
    print("ERROR: Could not find 'let stream = context.default_stream()'")
    raise SystemExit(1)

print(f"Found: new() at line {target_line+1}, Result at {result_line+1}, stream at {stream_line+1}")

# Body = everything from after ') -> Result<Self> {' to before 'let stream'
body_lines = lines[result_line + 1 : stream_line]

# Build replacement
replacement = [
    '    pub fn new(\n',
    '        context: Arc<CudaContext>,\n',
    '        topology: &PrismPrepTopology,\n',
    '        grid_dim: usize,\n',
    '        grid_spacing: f32,\n',
    '    ) -> Result<Self> {\n',
    '        let stream = context.default_stream();\n',
    '        Self::new_on_stream(context, stream, topology, grid_dim, grid_spacing)\n',
    '    }\n',
    '\n',
    '    /// Create new fused engine with explicit CUDA stream (for multi-stream concurrency).\n',
    '    pub fn new_on_stream(\n',
    '        context: Arc<CudaContext>,\n',
    '        stream: Arc<CudaStream>,\n',
    '        topology: &PrismPrepTopology,\n',
    '        grid_dim: usize,\n',
    '        grid_spacing: f32,\n',
    '    ) -> Result<Self> {\n',
] + body_lines  # includes log::info!, blank lines, if block — exact original whitespace

# Splice: replace lines[target_line : stream_line] with replacement
# stream_line itself gets kept (it becomes dead in new() but is the first line after body in new_on_stream)
# Actually we SKIP stream_line since new_on_stream doesn't need it (stream comes from param)
new_lines = lines[:target_line] + replacement + lines[stream_line + 1:]

with open(path, "w") as f:
    f.writelines(new_lines)

# Verify
with open(path, "r") as f:
    verify = f.read()

if 'new_on_stream' in verify and 'Self::new_on_stream' in verify:
    print(f"PATCH 1 applied correctly: {path}")
    # Show the result
    vlines = verify.split('\n')
    for i, l in enumerate(vlines):
        if 'new_on_stream' in l:
            start = max(0, i-2)
            end = min(len(vlines), i+3)
            for j in range(start, end):
                print(f"  {j+1}: {vlines[j]}")
            print("  ...")
            break
else:
    print("ERROR: Patch verification failed")
    raise SystemExit(1)
