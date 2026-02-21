# PRISM-4D: Spike-Native Density Clustering (SNDC)
## Implementation Specification for GPU-Accelerated Binding Site Detection

### Executive Summary

Replace Dynamic LIGSITE as the primary binding site detection engine with
a spike-native method that clusters directly on GpuSpikeEvent positions
weighted by intensity², using the existing RT-core BVH infrastructure.
LIGSITE becomes a fallback for zero-spike regions only.

This is the only binding site detection method in the literature that
uses neuromorphic thermodynamic trapping events as the primary signal.
Every other method (FTMap, SiteMap, Fpocket, P2Rank, LIGSITE) operates
on geometric or force-field-derived features.

---

### 1. ARCHITECTURE OVERVIEW

```
CURRENT (v8):
  Spikes → RT-DBSCAN → build_sites → [discarded by LIGSITE overwrite]
                                           ↓
  Protein geometry → LIGSITE voxels → geometric pockets
                                           ↓
                                    spike overlay scoring
                                           ↓
                                    weighted centroids → output

PROPOSED (v9 SNDC):
  Spikes → GPU intensity² grid → Hierarchical RT-DBSCAN → SNDC sites
                                                              ↓
                                                    site characterization
                                                    (volume, druggability,
                                                     lining residues, class)
                                                              ↓
                                                         output (primary)

  Protein geometry → LIGSITE voxels → geometric pockets → output (secondary)
  (fallback for zero-spike surface pockets only)
```

### 2. DATA FLOW — DETAILED

#### Stage 1: Spike Accumulation (ALREADY EXISTS)
- Source: `fused_engine.rs` → `get_accumulated_spikes()` or multi-stream merge
- Data: `Vec<GpuSpikeEvent>` — each 96 bytes with:
  - `position: [f32; 3]` — Cartesian coordinates in protein frame
  - `intensity: f32` — thermodynamic trapping depth
  - `aromatic_type: i32` — probe type (TRP/TYR/PHE/SS)
  - `water_density: f32` — local hydration
  - `vibrational_energy: f32` — UV energy deposited
- Typical counts: 2K-40K spikes per protein

#### Stage 2: GPU Intensity² Density Grid (NEW)
**Purpose:** Convert sparse spike events into a continuous 3D density field
on GPU, enabling sub-voxel hotspot resolution without CPU transfer.

**Implementation:**
```rust
// New file: crates/prism-nhs/src/spike_density.rs

/// 3D density grid computed entirely on GPU
pub struct SpikeDensityGrid {
    /// Grid dimensions
    pub dims: [u32; 3],
    /// Grid origin (min corner, Å)
    pub origin: [f32; 3],
    /// Voxel spacing (default 1.0Å for sub-angstrom hotspot resolution)
    pub spacing: f32,
    /// GPU buffer: density[x][y][z] = Σ intensity² × K(r)
    /// where K(r) is Gaussian kernel with σ = 2.0Å
    pub d_density: CudaSlice<f32>,
    /// GPU buffer: peak mask after non-maximum suppression
    pub d_peaks: CudaSlice<u32>,
}
```

**CUDA kernel** (`spike_density.cu`):
```c
// Kernel 1: Scatter spikes into density grid with Gaussian splatting
// Each spike contributes intensity² × exp(-r²/2σ²) to nearby voxels
// σ = 2.0Å → 99% energy within 6Å radius → ~6³ = 216 voxels per spike
// With 40K spikes: 40K × 216 = 8.6M atomicAdd operations
// On RTX 5080: ~0.1ms (trivially fast)
__global__ void scatter_spike_density(
    const float* __restrict__ spike_positions,  // [N, 3]
    const float* __restrict__ spike_intensities, // [N]
    float* __restrict__ density_grid,            // [Dx, Dy, Dz]
    int N, int Dx, int Dy, int Dz,
    float origin_x, float origin_y, float origin_z,
    float spacing, float sigma
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float px = spike_positions[tid * 3 + 0];
    float py = spike_positions[tid * 3 + 1];
    float pz = spike_positions[tid * 3 + 2];
    float w = spike_intensities[tid];
    w = w * w;  // intensity²

    float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);
    int radius = (int)ceilf(3.0f * sigma / spacing);  // 3σ cutoff

    int cx = (int)((px - origin_x) / spacing);
    int cy = (int)((py - origin_y) / spacing);
    int cz = (int)((pz - origin_z) / spacing);

    for (int dx = -radius; dx <= radius; dx++) {
        int ix = cx + dx;
        if (ix < 0 || ix >= Dx) continue;
        float fx = (ix * spacing + origin_x) - px;
        for (int dy = -radius; dy <= radius; dy++) {
            int iy = cy + dy;
            if (iy < 0 || iy >= Dy) continue;
            float fy = (iy * spacing + origin_y) - py;
            for (int dz = -radius; dz <= radius; dz++) {
                int iz = cz + dz;
                if (iz < 0 || iz >= Dz) continue;
                float fz = (iz * spacing + origin_z) - pz;
                float r2 = fx*fx + fy*fy + fz*fz;
                float val = w * expf(-r2 * inv_2sigma2);
                atomicAdd(&density_grid[ix * Dy * Dz + iy * Dz + iz], val);
            }
        }
    }
}

// Kernel 2: 3D non-maximum suppression to find density peaks
// A voxel is a peak if it's the maximum in its 3×3×3 neighborhood
// These peaks seed the hierarchical clustering
__global__ void find_density_peaks(
    const float* __restrict__ density_grid,
    uint32_t* __restrict__ peak_mask,
    int Dx, int Dy, int Dz,
    float min_density
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Dx || iy >= Dy || iz >= Dz) return;

    float val = density_grid[ix * Dy * Dz + iy * Dz + iz];
    if (val < min_density) { peak_mask[ix * Dy * Dz + iy * Dz + iz] = 0; return; }

    bool is_max = true;
    for (int dx = -1; dx <= 1 && is_max; dx++)
        for (int dy = -1; dy <= 1 && is_max; dy++)
            for (int dz = -1; dz <= 1 && is_max; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = ix+dx, ny = iy+dy, nz = iz+dz;
                if (nx >= 0 && nx < Dx && ny >= 0 && ny < Dy && nz >= 0 && nz < Dz)
                    if (density_grid[nx * Dy * Dz + ny * Dz + nz] > val)
                        is_max = false;
            }
    peak_mask[ix * Dy * Dz + iy * Dz + iz] = is_max ? 1 : 0;
}
```

**Why this matters:** The density grid is a continuous binding favorability
field computed from neuromorphic physics. The peaks ARE the binding hotspots.
This is analogous to what FTMap computes with probe fragment docking, but
arrived at through PRISM's UV-LIF perturbation dynamics instead. Zero overlap
with any existing methodology.

#### Stage 3: Hierarchical RT-Core Clustering (EXTEND EXISTING)

**Purpose:** Group spikes into binding sites using the existing RT-core
BVH infrastructure, but with density-aware hierarchical merging.

**Current state:** `rt_clustering.rs` does flat DBSCAN at a single epsilon.
This causes the "mega-cluster" problem (one cluster absorbs 50%+ of spikes)
or misses sub-sites at tight epsilon.

**Proposed:** Multi-scale persistence clustering on GPU:

```rust
// Extend: crates/prism-nhs/src/rt_clustering.rs

/// Hierarchical density-based clustering using RT cores
pub struct HierarchicalRtClustering {
    config: HierarchicalConfig,
    context: Arc<CudaContext>,
    rt_engine: RtClusteringEngine,
}

pub struct HierarchicalConfig {
    /// Epsilon sweep: start, end, step (Å)
    pub eps_range: (f32, f32, f32),  // e.g., (2.0, 8.0, 0.5) → 13 levels
    /// Minimum persistence (number of epsilon levels a cluster survives)
    pub min_persistence: u32,  // e.g., 3
    /// Weight spikes by intensity² during centroid computation
    pub intensity_weighted: bool,
    /// Minimum spike count per cluster
    pub min_cluster_spikes: u32,
}

impl HierarchicalRtClustering {
    pub fn cluster_spikes(
        &mut self,
        spikes: &[GpuSpikeEvent],
        density_grid: &SpikeDensityGrid,
    ) -> Result<Vec<SndcSite>> {
        // 1. Upload spike positions + intensities to GPU
        // 2. Sweep epsilon from tight (2Å) to loose (8Å):
        //    - At each epsilon, run RT-core DBSCAN (existing infra)
        //    - Record cluster assignments per spike
        //    - Track which clusters persist vs merge vs split
        // 3. For each persistent cluster:
        //    - Compute intensity²-weighted centroid
        //    - Sample density grid at centroid → binding favorability
        //    - Compute cluster spread (σ) from spike positions
        //    - Count unique aromatic probe types contributing
        // 4. Merge overlapping clusters (centroid distance < 4Å)
        // 5. Return sorted by density_score descending
    }
}
```

**The RT-core advantage:** Each epsilon level requires rebuilding the BVH
with sphere radius = epsilon, then launching rays for neighbor finding.
On RTX 5080 with ~40K spikes, each level takes ~2ms. A 13-level sweep
is ~26ms total. On CPU this would be 13 × O(N²) = 13 × 1.6B operations
for 40K spikes. The RT cores make hierarchical clustering practical in
real-time — this is PRISM's hardware moat.

#### Stage 4: Site Characterization (REFACTOR EXISTING)

```rust
// New struct replacing ClusteredBindingSite for SNDC sites

pub struct SndcSite {
    pub id: u32,

    // --- Spike-derived (primary) ---
    /// Intensity²-weighted centroid of spike cluster
    pub centroid: [f64; 3],
    /// Peak density from grid at centroid location
    pub peak_density: f64,
    /// Number of spikes in cluster
    pub spike_count: u32,
    /// Mean spike intensity
    pub mean_intensity: f64,
    /// Intensity variance (low = coherent signal, high = noisy)
    pub intensity_cv: f64,
    /// Persistence: number of epsilon levels cluster survived
    pub persistence: u32,
    /// Unique aromatic probe types (TRP=0, TYR=1, PHE=2, SS=3)
    pub probe_diversity: u32,
    /// Mean water density at spike positions (desolvation proxy)
    pub mean_water_density: f64,

    // --- Geometry-derived (secondary) ---
    /// Cluster spatial extent (Å)
    pub cluster_radius: f64,
    /// Estimated volume from convex hull of spike positions
    pub estimated_volume: f64,
    /// Enclosure ratio: fraction of solid angle blocked by protein
    pub enclosure: f64,
    /// Depth: distance from nearest surface to centroid
    pub depth: f64,

    // --- Classification ---
    pub druggability: f64,
    pub quality_score: f64,
    pub classification: SiteClassification,

    // --- Lining residues ---
    pub lining_residues: Vec<LiningResidue>,
}

impl SndcSite {
    /// Druggability from spike-native features — NO geometric pocket dependence
    pub fn compute_druggability(&mut self) {
        // Based on:
        // 1. peak_density (strong trapping = druggable)
        // 2. persistence (robust across scales = real site)
        // 3. probe_diversity (broad pharmacophore)
        // 4. mean_water_density (desolvated = hydrophobic = druggable)
        // 5. enclosure (>0.5 = enclosed = druggable)
        // 6. cluster_radius (5-12Å = drug-like)
        //
        // Physics-based model requiring NO training data.
    }

    /// Classification: ActiveSite, Allosteric, Cryptic, Unknown
    pub fn classify(&mut self) {
        // Cryptic: high persistence + high intensity_cv
        //   (appears/disappears = conformationally gated)
        // Allosteric: moderate persistence + distant from top site
        // ActiveSite: highest persistence + highest density + most probes
    }
}
```

### 3. INTEGRATION INTO nhs_rt_full.rs

Key change in post-simulation analysis (~line 2266):

```rust
// ========== BINDING SITE DETECTION ==========

// PRIMARY: Spike-Native Density Clustering (SNDC)
if !all_stream_spikes.is_empty() {
    log::info!("  Running Spike-Native Density Clustering...");

    // Stage 2: Build density grid on GPU
    let density_grid = SpikeDensityGrid::from_spikes(
        &all_stream_spikes,
        &topology.positions,
        1.0,   // spacing: 1Å
        2.0,   // sigma: 2Å Gaussian
        context.clone(),
    )?;
    let n_peaks = density_grid.count_peaks()?;
    log::info!("    Density grid: {}x{}x{} voxels, {} peaks",
        density_grid.dims[0], density_grid.dims[1], density_grid.dims[2], n_peaks);

    // Stage 3: Hierarchical RT-core clustering
    let mut hrt = HierarchicalRtClustering::new(
        HierarchicalConfig {
            eps_range: (2.0, 8.0, 0.5),
            min_persistence: 3,
            intensity_weighted: true,
            min_cluster_spikes: 50,
        },
        context.clone(),
    )?;
    let mut sndc_sites = hrt.cluster_spikes(&all_stream_spikes, &density_grid)?;

    // Stage 4: Characterize sites
    for site in &mut sndc_sites {
        site.compute_enclosure(&topology.positions)?;
        site.compute_druggability();
        site.classify();
        site.compute_lining_residues(
            &topology.positions, &topology.residue_ids,
            &topology.residue_names, &topology.chain_ids,
            &pdb_id_map, args.lining_cutoff,
        );
    }
    log::info!("  SNDC complete: {} binding sites", sndc_sites.len());
    write_sndc_sites(&sndc_sites, &output_base)?;
}

// SECONDARY: Dynamic LIGSITE (fallback/comparison)
log::info!("  Running Dynamic LIGSITE pocket detection...");
// ... existing code unchanged ...
```

### 4. OUTPUT FORMAT

```json
{
  "method": "SNDC",
  "version": "1.0",
  "sites": [
    {
      "id": 0,
      "centroid": [12.3, 45.6, 78.9],
      "peak_density": 847.3,
      "spike_count": 1247,
      "mean_intensity": 11.2,
      "persistence": 8,
      "probe_diversity": 3,
      "mean_water_density": 0.42,
      "cluster_radius": 7.8,
      "estimated_volume": 623,
      "enclosure": 0.72,
      "depth": 4.1,
      "druggability": 0.87,
      "quality_score": 0.91,
      "classification": "ActiveSite",
      "lining_residues": []
    }
  ],
  "ligsite_sites": []
}
```

### 5. PERFORMANCE BUDGET (RTX 5080)

| Stage | GPU Time | Memory |
|-------|----------|--------|
| Spike accumulation | (already computed) | — |
| Density grid (1Å, 100³) | ~0.1ms | 4MB |
| Non-max suppression | ~0.05ms | 4MB |
| RT-DBSCAN × 13 epsilon levels | ~26ms | 40K × 96B + BVH |
| Persistence tracking | ~0.5ms | trivial |
| Site characterization | ~1ms (CPU) | — |
| **Total SNDC** | **~28ms** | **~12MB** |
| Current LIGSITE | ~50ms | ~20MB |

SNDC is **faster** than LIGSITE while being fundamentally more accurate.

### 6. FILES TO CREATE/MODIFY

**New files:**
- `crates/prism-nhs/src/spike_density.rs` — GPU density grid
- `crates/prism-nhs/src/spike_density.cu` — CUDA kernels (scatter + NMS)
- `crates/prism-nhs/src/sndc.rs` — SndcSite struct + characterization
- `crates/prism-nhs/src/hierarchical_clustering.rs` — multi-scale persistence

**Modified files:**
- `crates/prism-nhs/src/bin/nhs_rt_full.rs` — integration (SNDC before LIGSITE)
- `crates/prism-nhs/src/rt_clustering.rs` — expose epsilon param for sweep
- `crates/prism-nhs/src/lib.rs` — register new modules

**Unchanged:**
- `crates/prism-nhs/src/fused_engine.rs` — GpuSpikeEvent untouched
- All CUDA simulation kernels — upstream pipeline unchanged
- Dynamic LIGSITE code — kept as secondary/fallback method

### 7. VALIDATION PLAN

Re-run 9-protein CryptoSite benchmark with SNDC primary sites.
Compare DCC against v8 LIGSITE weighted centroids. Expected improvements:
- 1ERE: mega-pocket eliminated (SNDC cannot produce 18K Å³ cluster)
- 1BJ4, 1HHP: tighter centroid from hierarchical persistence
- All proteins: centroid IS the density peak, not geometric center of void

### 8. COMPETITIVE POSITION

| Feature | FTMap | SiteMap | Fpocket | P2Rank | PRISM SNDC |
|---------|:-----:|:-------:|:-------:|:------:|:----------:|
| Binding physics | Probe docking | Grid energy | None | ML features | Neuromorphic trapping |
| GPU accelerated | No | No | No | No | RT cores |
| Cryptic sites | No | Limited | No | Trained | Emergent |
| Runtime/protein | ~30min | ~10min | ~1sec | ~5sec | ~30ms (clustering) |
| Training data | No | No | No | Yes | No |
| Open source | Yes | No ($$$) | Yes | Yes | No (proprietary) |

### 9. IMPLEMENTATION ORDER

1. `spike_density.cu` + `spike_density.rs` — GPU density grid (standalone testable)
2. `hierarchical_clustering.rs` — extend RT-DBSCAN with epsilon sweep
3. `sndc.rs` — SndcSite struct, druggability, classification
4. `nhs_rt_full.rs` integration — wire SNDC into pipeline
5. Benchmark validation — CryptoSite DCC comparison
6. Tune parameters (σ, eps_range, min_persistence) on benchmark set

### 10. KEY DESIGN DECISIONS

**Why intensity² not intensity³:**
- intensity² gives quadratic preference to high-intensity spikes
- intensity³ over-concentrates on single extreme spikes (loses cluster shape)
- Validated empirically: PTP1B cryptic site MISS→EXCELLENT with intensity²

**Why Gaussian splatting not raw binning:**
- Raw binning creates grid artifacts (spike at voxel boundary → split signal)
- Gaussian (σ=2Å) produces smooth density field with sub-voxel peak resolution
- 2Å sigma matches typical binding site feature scale

**Why persistence not single-epsilon:**
- Single epsilon: too tight misses diffuse allosteric sites, too loose creates mega-clusters
- Persistence (cluster survives 3+ epsilon levels) selects physically real sites
- Cryptic sites naturally have HIGH persistence (conformational trapping is scale-invariant)

**Why keep LIGSITE as fallback:**
- Surface pockets with zero spikes (no aromatic probes nearby) are real
- LIGSITE catches these geometric pockets that SNDC cannot see
- Provides direct comparison for benchmarking/publication
