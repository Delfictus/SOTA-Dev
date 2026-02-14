#!/usr/bin/env python3
"""
PRISM-4D Pharmacophore Hotspot Map Generator
Produces OpenDX (.dx) grids from spike events for PyMOL/ChimeraX visualization.
Also extracts the centroid (open-state) frame from ensemble trajectories.
"""
import json
import numpy as np
import sys
import os
import glob

def load_spike_events(path):
    with open(path) as f:
        return json.load(f)

def build_density_grid(spikes, grid_spacing=1.0, sigma=1.5):
    """Build 3D density grid from spike positions using Gaussian kernel."""
    positions = np.array([[s['x'], s['y'], s['z']] for s in spikes])
    intensities = np.array([s['intensity'] for s in spikes])
    
    pad = 4.0
    mins = positions.min(axis=0) - pad
    maxs = positions.max(axis=0) + pad
    
    nx = int(np.ceil((maxs[0] - mins[0]) / grid_spacing)) + 1
    ny = int(np.ceil((maxs[1] - mins[1]) / grid_spacing)) + 1
    nz = int(np.ceil((maxs[2] - mins[2]) / grid_spacing)) + 1
    
    grid = np.zeros((nx, ny, nz), dtype=np.float32)
    
    cutoff = int(3 * sigma / grid_spacing) + 1
    for pos, intensity in zip(positions, intensities):
        ix = int((pos[0] - mins[0]) / grid_spacing)
        iy = int((pos[1] - mins[1]) / grid_spacing)
        iz = int((pos[2] - mins[2]) / grid_spacing)
        
        for dx in range(-cutoff, cutoff + 1):
            for dy in range(-cutoff, cutoff + 1):
                for dz in range(-cutoff, cutoff + 1):
                    gx, gy, gz = ix + dx, iy + dy, iz + dz
                    if 0 <= gx < nx and 0 <= gy < ny and 0 <= gz < nz:
                        dist2 = (dx * grid_spacing)**2 + (dy * grid_spacing)**2 + (dz * grid_spacing)**2
                        grid[gx, gy, gz] += intensity * np.exp(-dist2 / (2 * sigma**2))
    
    return grid, mins, (nx, ny, nz), grid_spacing

def write_dx(grid, origin, dims, spacing, filepath, comment="PRISM-4D hotspot"):
    """Write OpenDX format grid file."""
    nx, ny, nz = dims
    with open(filepath, 'w') as f:
        f.write(f"# {comment}\n")
        f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        f.write(f"origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
        f.write(f"delta {spacing:.6f} 0.000000 0.000000\n")
        f.write(f"delta 0.000000 {spacing:.6f} 0.000000\n")
        f.write(f"delta 0.000000 0.000000 {spacing:.6f}\n")
        f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        f.write(f"object 3 class array type double rank 0 items {nx*ny*nz} data follows\n")
        
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f"{grid[ix, iy, iz]:.6e}")
                    count += 1
                    if count % 3 == 0:
                        f.write("\n")
                    else:
                        f.write(" ")
        if count % 3 != 0:
            f.write("\n")
        
        f.write('attribute "dep" string "positions"\n')
        f.write('object "regular positions regular connections" class field\n')
        f.write('component "positions" value 1\n')
        f.write('component "connections" value 2\n')
        f.write('component "data" value 3\n')

def find_open_frame(spikes, centroid):
    """Find the frame with most spikes near centroid = most open pocket state."""
    cx, cy, cz = centroid
    frame_scores = {}
    for s in spikes:
        fi = s['frame_index']
        dx = s['x'] - cx
        dy = s['y'] - cy
        dz = s['z'] - cz
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 6.0:
            frame_scores[fi] = frame_scores.get(fi, 0) + s['intensity']
    
    if not frame_scores:
        return 0
    best = max(frame_scores, key=frame_scores.get)
    print(f"    Frame scores (top 5):")
    for fi, score in sorted(frame_scores.items(), key=lambda x: -x[1])[:5]:
        marker = " <<<" if fi == best else ""
        print(f"      Frame {fi:2d}: {score:.1f}{marker}")
    return best

def extract_frame_pdb(traj_dir, structure_name, stream_id, frame_idx, output_path):
    """Extract a specific frame from ensemble trajectory PDB."""
    stem = structure_name.replace('.topology', '')
    traj_path = os.path.join(traj_dir, f"{stem}_stream{stream_id:02d}.ensemble.pdb")
    
    if not os.path.exists(traj_path):
        candidates = glob.glob(os.path.join(traj_dir, f"{stem}*ensemble*.pdb"))
        if candidates:
            traj_path = candidates[0]
        else:
            print(f"  Warning: No trajectory found at {traj_path}")
            return False
    
    models = []
    current_model = []
    in_model = False
    with open(traj_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                in_model = True
                current_model = [line]
            elif line.startswith('ENDMDL'):
                current_model.append(line)
                models.append(current_model)
                in_model = False
            elif in_model:
                current_model.append(line)
    
    if frame_idx < len(models):
        with open(output_path, 'w') as f:
            f.write(f"REMARK PRISM-4D open-state frame {frame_idx} from stream {stream_id}\n")
            f.write(f"REMARK Extracted as highest spike-density frame (most open pocket)\n")
            for line in models[frame_idx]:
                if not line.startswith('MODEL') and not line.startswith('ENDMDL'):
                    f.write(line)
            f.write("END\n")
        return True
    else:
        print(f"  Warning: Frame {frame_idx} not found (only {len(models)} models)")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: pharmacophore_extract.py <spike_events.json> <output_dir> [traj_dir]")
        sys.exit(1)
    
    spike_path = sys.argv[1]
    output_dir = sys.argv[2]
    traj_dir = sys.argv[3] if len(sys.argv) > 3 else output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading spike events...")
    data = load_spike_events(spike_path)
    spikes = data['spikes']
    centroid = data['centroid']
    site_id = data['site_id']
    
    print(f"\nPRISM-4D Pharmacophore Extraction")
    print(f"  Site {site_id}: {len(spikes)} spikes at centroid [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]")
    
    # Type breakdown
    from collections import Counter
    types = Counter(s['type'] for s in spikes)
    sources = Counter(s['spike_source'] for s in spikes)
    print(f"  Types: {dict(types)}")
    print(f"  Sources: {dict(sources)}")
    
    # === 1. Combined hotspot map ===
    print("\n  [1/4] Building combined hotspot map...")
    grid, origin, dims, spacing = build_density_grid(spikes, grid_spacing=1.0, sigma=1.5)
    combined_path = os.path.join(output_dir, f"site{site_id}_combined.dx")
    write_dx(grid, origin, dims, spacing, combined_path, "PRISM-4D combined hotspot")
    grid_max = grid.max()
    print(f"    -> {combined_path} ({dims[0]}x{dims[1]}x{dims[2]}, max={grid_max:.1f})")
    
    # === 2. Per-type hotspot maps ===
    print("\n  [2/4] Building per-type hotspot maps...")
    type_map = {}
    for s in spikes:
        t = s['type']
        if t not in type_map:
            type_map[t] = []
        type_map[t].append(s)
    
    type_maxes = {}
    for spike_type, type_spikes in sorted(type_map.items()):
        if len(type_spikes) < 10:
            continue
        print(f"    {spike_type}: {len(type_spikes)} spikes...")
        tgrid, torigin, tdims, tspacing = build_density_grid(type_spikes, grid_spacing=1.0, sigma=1.5)
        dx_path = os.path.join(output_dir, f"site{site_id}_{spike_type.lower()}.dx")
        write_dx(tgrid, torigin, tdims, tspacing, dx_path, f"PRISM-4D {spike_type} hotspot")
        type_maxes[spike_type] = tgrid.max()
        print(f"      -> {dx_path} (max={tgrid.max():.1f})")
    
    # === 3. Find open-state frame ===
    print(f"\n  [3/4] Finding open-state (max density) frame...")
    open_frame = find_open_frame(spikes, centroid)
    print(f"    Best frame: {open_frame}")
    
    basename = os.path.basename(spike_path)
    structure_name = basename.split('.site')[0]
    
    open_pdb_path = os.path.join(output_dir, f"{structure_name}_open_frame{open_frame}.pdb")
    if extract_frame_pdb(traj_dir, structure_name, 0, open_frame, open_pdb_path):
        print(f"    -> {open_pdb_path}")
    
    # === 4. PyMOL visualization script ===
    print(f"\n  [4/4] Writing PyMOL script...")
    pymol_script = os.path.join(output_dir, f"site{site_id}_visualize.pml")
    with open(pymol_script, 'w') as f:
        f.write(f"""# PRISM-4D Site {site_id} Pharmacophore Visualization
# Load in PyMOL: @site{site_id}_visualize.pml

# Load receptor (open-state frame)
load {os.path.basename(open_pdb_path)}, receptor
color gray80, receptor
show cartoon, receptor
hide lines, receptor

# Load combined hotspot
load site{site_id}_combined.dx, combined_hotspot
isosurface combined_surf, combined_hotspot, {grid_max * 0.3:.1f}
color red, combined_surf
set transparency, 0.4, combined_surf

""")
        colors = {'bnz': 'orange', 'tyr': 'cyan', 'trp': 'magenta', 'phe': 'yellow', 'unk': 'blue'}
        for spike_type in sorted(type_map.keys()):
            if len(type_map[spike_type]) < 10:
                continue
            dx_name = f"site{site_id}_{spike_type.lower()}"
            color = colors.get(spike_type.lower(), 'white')
            tmax = type_maxes.get(spike_type, grid_max)
            f.write(f"# {spike_type} hotspot ({len(type_map[spike_type])} spikes)\n")
            f.write(f"load {dx_name}.dx, {dx_name}\n")
            f.write(f"isosurface {dx_name}_surf, {dx_name}, {tmax * 0.25:.1f}\n")
            f.write(f"color {color}, {dx_name}_surf\n")
            f.write(f"set transparency, 0.5, {dx_name}_surf\n\n")
        
        f.write(f"""# Centroid marker
pseudoatom centroid, pos=[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]
show spheres, centroid
color red, centroid
set sphere_scale, 0.5, centroid

# Show lining residues as sticks
select pocket, receptor within 8.0 of centroid
show sticks, pocket
color palegreen, pocket and elem C

# Center view
center centroid
zoom centroid, 15
set ray_shadows, 1
set ray_trace_mode, 1
bg_color white
""")
    
    print(f"    -> {pymol_script}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  PRISM-4D Pharmacophore Extraction Complete")
    print(f"{'='*60}")
    print(f"  Site {site_id} @ [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]")
    print(f"  Total spikes: {len(spikes)}")
    print(f"  Open-state frame: {open_frame}")
    print(f"  Files generated:")
    print(f"    {combined_path}")
    for spike_type in sorted(type_map.keys()):
        if len(type_map[spike_type]) >= 10:
            print(f"    {output_dir}/site{site_id}_{spike_type.lower()}.dx")
    print(f"    {open_pdb_path}")
    print(f"    {pymol_script}")
    print(f"\n  Open in PyMOL:")
    print(f"    cd {output_dir}")
    print(f"    pymol @{os.path.basename(pymol_script)}")

if __name__ == '__main__':
    main()
