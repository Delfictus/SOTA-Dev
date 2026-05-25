#!/usr/bin/env python3
"""FULL-SPECTRUM NON-NAIVE DATA EXTRACTION
Mines every parquet, every column, every distribution.
Memory-safe: uses PyArrow batch iteration for the 106GB spike dataset.
"""
import pyarrow.parquet as pq
import polars as pl
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale")
RAW = Path("/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map")
TOPO = Path("/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/04_TOPOLOGIES")

def sep(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

# ════════════════════════════════════════════════════════════════
sep("1. PROTOCOL PHASE BOUNDARIES — ALL CONDITIONS")
# ════════════════════════════════════════════════════════════════
for cond_dir in sorted(RAW.iterdir()):
    if not cond_dir.is_dir():
        continue
    rep0 = cond_dir / "replica_0"
    if not rep0.exists():
        continue
    jsons = sorted(rep0.glob("*protocol_state*"))[:3]
    for f in jsons:
        data = json.loads(f.read_text())
        stream = data.get("stream_index", "?")
        print(f"{cond_dir.name} stream {stream}: "
              f"cold_end={data.get('cold_hold_end')} "
              f"ramp_end={data.get('ramp_end')} "
              f"warm_end={data.get('warm_hold_end')} "
              f"ramp_down_end={data.get('ramp_down_end')} "
              f"total={data.get('total_steps')} "
              f"dt={data.get('dt_ps',0):.6f} "
              f"uv_energy={data.get('uv_burst_energy_kcal_mol')} "
              f"uv_interval={data.get('uv_burst_interval_steps')} "
              f"uv_wl={data.get('uv_wavelength_nm')}")

# ════════════════════════════════════════════════════════════════
sep("2. SPIKE TEMPORAL STRUCTURE — FULL 5.62B DATASET")
# ════════════════════════════════════════════════════════════════
spike_path = str(BASE / "spike_events_snr_masked.parquet")
pf = pq.ParquetFile(spike_path)
n_groups = pf.metadata.num_row_groups
print(f"Row groups: {n_groups}, Total rows: {pf.metadata.num_rows:,}")

# Sample row groups across the full file to find true temporal span
print("\n--- Row group sampling (every 1000th) ---")
global_ts_min = float('inf')
global_ts_max = float('-inf')
global_time_min = float('inf')
global_time_max = float('-inf')
conditions_seen = set()
streams_seen = set()

step = max(1, n_groups // 60)
ts_bins_by_condition = defaultdict(Counter)
residue_first_ts = defaultdict(lambda: defaultdict(lambda: float('inf')))
residue_phase_counts = defaultdict(lambda: defaultdict(Counter))

for gi in range(0, n_groups, step):
    batch = pf.read_row_group(gi, columns=[
        'condition_id', 'primary_residue_idx', 'timestep',
        'time_ps', 'stream_id', 'intensity', 'causal_anchor',
        'wavelength_nm', 'phase_bits', 'spike_source',
        'water_density', 'wd_change', 'vibrational_energy'
    ])
    df = pl.from_arrow(batch)
    
    ts_min = df["timestep"].min()
    ts_max = df["timestep"].max()
    if ts_min < global_ts_min: global_ts_min = ts_min
    if ts_max > global_ts_max: global_ts_max = ts_max
    
    time_min = df["time_ps"].min()
    time_max = df["time_ps"].max()
    if time_min < global_time_min: global_time_min = time_min
    if time_max > global_time_max: global_time_max = time_max
    
    conditions_seen.update(df["condition_id"].unique().to_list())
    streams_seen.update(df["stream_id"].unique().to_list())
    
    # Bin timesteps per condition
    for cond in df["condition_id"].unique().to_list():
        cdf = df.filter(pl.col("condition_id") == cond)
        for ts in cdf.with_columns(
            (pl.col("timestep") // 2000 * 2000).alias("bin")
        )["bin"].to_list():
            ts_bins_by_condition[cond][ts] += 1
        
        # Track first timestep per residue per condition
        for row in cdf.group_by("primary_residue_idx").agg(
            pl.col("timestep").min().alias("min_ts")
        ).iter_rows():
            res_idx, min_ts = row
            if min_ts < residue_first_ts[cond][res_idx]:
                residue_first_ts[cond][res_idx] = min_ts

print(f"\nGLOBAL timestep range: {global_ts_min} to {global_ts_max}")
print(f"GLOBAL time_ps range: {global_time_min:.4f} to {global_time_max:.4f}")
print(f"Conditions: {sorted(conditions_seen)}")
print(f"Streams: {sorted(streams_seen)}")

# Print temporal histograms per condition
for cond in sorted(ts_bins_by_condition.keys()):
    bins = ts_bins_by_condition[cond]
    print(f"\n--- {cond} temporal histogram ---")
    for k in sorted(bins.keys()):
        bar = "#" * min(50, bins[k] // max(1, max(bins.values()) // 50))
        print(f"  {k:>8} | {bins[k]:>10,} {bar}")

# ════════════════════════════════════════════════════════════════
sep("3. PER-RESIDUE FIRST SPIKE DURING RAMP (temporal cascade)")
# ════════════════════════════════════════════════════════════════
# For glp1r_5VEX_WT, ramp starts at step 10000
# Show which residues fire FIRST after step 10000
for cond in ["glp1r_5VEX_WT"]:
    if cond not in residue_first_ts:
        print(f"No data for {cond}")
        continue
    rft = residue_first_ts[cond]
    # Get residues whose first spike is AFTER cold_hold
    post_cold = {r: t for r, t in rft.items() if t > 5000}
    if post_cold:
        sorted_res = sorted(post_cold.items(), key=lambda x: x[1])
        print(f"{cond}: {len(post_cold)} residues with first spike after step 5000")
        print("  Top 20 LATEST first-spike residues:")
        for r, t in sorted_res[-20:]:
            print(f"    residue {r:>4}: first spike at step {t}")
    else:
        print(f"{cond}: ALL residues fire before step 5000")
    
    # Distribution of first-spike timesteps
    all_firsts = list(rft.values())
    print(f"\n  First-spike distribution: min={min(all_firsts)}, "
          f"max={max(all_firsts)}, mean={np.mean(all_firsts):.0f}, "
          f"median={np.median(all_firsts):.0f}")
    # Histogram of first-spike times
    hist, edges = np.histogram(all_firsts, bins=10)
    for i in range(len(hist)):
        print(f"    {edges[i]:>8.0f}-{edges[i+1]:>8.0f}: {hist[i]:>4} residues")

# ════════════════════════════════════════════════════════════════
sep("4. SPIKE COLUMN DEEP DIVE — phase_bits, spike_source, wavelength")
# ════════════════════════════════════════════════════════════════
# Sample a larger batch to understand these columns
batch = pf.read_row_group(n_groups // 2, columns=[
    'condition_id', 'phase_bits', 'spike_source', 'wavelength_nm',
    'intensity', 'water_density', 'wd_change', 'vibrational_energy',
    'causal_anchor', 'primary_residue_idx', 'timestep'
])
df = pl.from_arrow(batch)
print(f"Sample size: {df.shape[0]:,} from middle row group")

for col in ['phase_bits', 'spike_source', 'wavelength_nm']:
    print(f"\n{col}: nunique={df[col].n_unique()}, "
          f"min={df[col].min()}, max={df[col].max()}")
    vc = df[col].value_counts().sort("count", descending=True).head(10)
    print(vc)

print("\n--- intensity distribution ---")
print(f"min={df['intensity'].min():.2f}, max={df['intensity'].max():.2f}, "
      f"mean={df['intensity'].mean():.2f}, std={df['intensity'].std():.2f}")

print("\n--- water_density distribution ---")
print(f"min={df['water_density'].min():.4f}, max={df['water_density'].max():.4f}, "
      f"mean={df['water_density'].mean():.4f}")

print("\n--- wd_change distribution ---")
print(f"min={df['wd_change'].min():.4f}, max={df['wd_change'].max():.4f}, "
      f"mean={df['wd_change'].mean():.4f}")

print("\n--- vibrational_energy distribution ---")
print(f"min={df['vibrational_energy'].min():.4f}, max={df['vibrational_energy'].max():.4f}, "
      f"mean={df['vibrational_energy'].mean():.4f}")

print("\n--- causal_anchor breakdown ---")
print(df['causal_anchor'].value_counts())

# ════════════════════════════════════════════════════════════════
sep("5. KCC DEEP DIVE — ALL non-NaN columns")
# ════════════════════════════════════════════════════════════════
kcc = pl.read_parquet(str(BASE / "kcc_residue_fields.parquet"))
print(f"Rows: {kcc.shape[0]}, Columns: {kcc.columns}")

for col in kcc.columns:
    if kcc[col].dtype in [pl.Float64, pl.Float32]:
        nans = kcc.filter(pl.col(col).is_nan()).shape[0]
        if nans == kcc.shape[0]:
            print(f"  {col}: ALL NaN")
        else:
            valid = kcc.filter(~pl.col(col).is_nan())[col]
            print(f"  {col}: valid={valid.len()}, min={valid.min():.6f}, "
                  f"max={valid.max():.6f}, mean={valid.mean():.6f}, "
                  f"std={valid.std():.6f}")
    elif kcc[col].dtype in [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
        print(f"  {col}: min={kcc[col].min()}, max={kcc[col].max()}, "
              f"mean={kcc[col].mean():.2f}")

# Per-residue temporal_corr distribution for critical edge residues
rm = pl.read_parquet(str(BASE / "receptor_durability_risk_map.parquet"))
if "edge_from_residue" in rm.columns:
    edge_res = set(rm["edge_from_residue"].to_list() + rm["edge_to_residue"].to_list())
    kcc_edges = kcc.filter(pl.col("residue_idx").is_in(list(edge_res)))
    print(f"\nKCC for {len(edge_res)} critical edge residues: {kcc_edges.shape[0]} rows")
    if kcc_edges.shape[0] > 0:
        print(kcc_edges.group_by("residue_idx").agg([
            pl.col("temporal_corr").mean().alias("mean_tc"),
            pl.col("temporal_corr").std().alias("std_tc"),
            pl.col("direction_score").mean().alias("mean_dir"),
            pl.col("motion_efficiency").mean().alias("mean_eff"),
            pl.col("active_causal").mean().alias("mean_active"),
        ]).sort("mean_tc", descending=True))

# ════════════════════════════════════════════════════════════════
sep("6. BOCPD DEEP DIVE — regime transitions vs protocol phase")
# ════════════════════════════════════════════════════════════════
bocpd = pl.read_parquet(str(BASE / "bocpd_survival_regimes.parquet"))
print(f"Rows: {bocpd.shape[0]}, Columns: {bocpd.columns}")

print("\n--- Regime transitions by thermal_phase ---")
print(bocpd["thermal_phase"].value_counts().sort("count", descending=True))

print("\n--- Regime transitions by condition ---")
print(bocpd.group_by("condition_id").agg([
    pl.len().alias("n_regimes"),
    pl.col("survival_time_ps").mean().alias("mean_survival"),
    pl.col("frame_idx").max().alias("max_frame"),
    pl.col("reset_probability").mean().alias("mean_reset_prob"),
    pl.col("posterior_max").mean().alias("mean_posterior"),
]).sort("condition_id"))

print("\n--- frame_idx distribution ---")
print(f"min={bocpd['frame_idx'].min()}, max={bocpd['frame_idx'].max()}, "
      f"nunique={bocpd['frame_idx'].n_unique()}")
print(bocpd["frame_idx"].value_counts().sort("frame_idx").head(20))

# ════════════════════════════════════════════════════════════════
sep("7. STEERING TENSOR — phase-specific targeting")
# ════════════════════════════════════════════════════════════════
st = pl.read_parquet(str(BASE / "autonomous_steering_tensor.parquet"))
print(f"Rows: {st.shape[0]}, Columns: {st.columns}")

# Which residues have highest steering weight across ALL conditions?
top_steered = (
    st.group_by("residue_idx")
    .agg([
        pl.col("steering_weight_mean").mean().alias("global_mean_weight"),
        pl.col("condition_id").n_unique().alias("n_conditions"),
        pl.col("thermal_phase").n_unique().alias("n_phases"),
        pl.col("supporting_stream_count").mean().alias("mean_streams"),
    ])
    .sort("global_mean_weight", descending=True)
    .head(20)
)
print("\nTop 20 globally steered residues:")
print(top_steered)

# Phase-specific steering: do different phases target different residues?
print("\n--- Top 5 steered residues per thermal phase ---")
for phase in st["thermal_phase"].unique().sort().to_list():
    phase_top = (
        st.filter(pl.col("thermal_phase") == phase)
        .group_by("residue_idx")
        .agg(pl.col("steering_weight_mean").mean().alias("mean_w"))
        .sort("mean_w", descending=True)
        .head(5)
    )
    print(f"\n{phase}:")
    print(phase_top)

# ════════════════════════════════════════════════════════════════
sep("8. AROMATIC TENSOR — displacement at critical edges")
# ════════════════════════════════════════════════════════════════
arom = pl.read_parquet(str(BASE / "aromatic_reorganization_tensor.parquet"))
print(f"Rows: {arom.shape[0]}, Columns: {arom.columns}")

# Compute total displacement magnitude
if "centroid_displacement_x" in arom.columns:
    arom_enriched = arom.with_columns(
        (pl.col("centroid_displacement_x")**2 + 
         pl.col("centroid_displacement_y")**2 + 
         pl.col("centroid_displacement_z")**2).sqrt().alias("displacement_magnitude")
    )
    print("\n--- Displacement magnitude distribution ---")
    dm = arom_enriched["displacement_magnitude"]
    print(f"min={dm.min():.6f}, max={dm.max():.6f}, mean={dm.mean():.6f}, std={dm.std():.6f}")
    
    print("\nTop 15 most displaced aromatics:")
    print(arom_enriched.sort("displacement_magnitude", descending=True).head(15).select([
        "condition_id", "ring_idx", "displacement_magnitude",
        "centroid_displacement_std", "cold_stream_count", "warm_stream_count"
    ]))

# ════════════════════════════════════════════════════════════════
sep("9. MECHANICAL LOAD — per-stream variation (not just aggregate)")
# ════════════════════════════════════════════════════════════════
mech = pl.scan_parquet(str(BASE / "mechanical_load_network.parquet"))
# For ASN182 (the wire residue), show per-stream load variation
asn_loads = (
    mech.filter(pl.col("residue_idx").is_in([46, 144, 148]))  # ASN182 indices
    .group_by(["condition_id", "residue_idx", "stream_id"])
    .agg([
        pl.col("mechanical_load").mean().alias("mean_load"),
        pl.col("mechanical_load").std().alias("std_load"),
        pl.len().alias("n_atoms"),
    ])
    .collect()
)

if asn_loads.shape[0] > 0:
    print("ASN182 (wire residue) per-stream load variation:")
    for cond in ["glp1r_5VEX_WT", "glp1r_6XOX_WT"]:
        subset = asn_loads.filter(pl.col("condition_id") == cond)
        if subset.shape[0] > 0:
            print(f"\n  {cond}:")
            print(f"    streams: {subset.shape[0]}")
            print(f"    load range: {subset['mean_load'].min():.1f} to {subset['mean_load'].max():.1f}")
            print(f"    load mean: {subset['mean_load'].mean():.1f} ± {subset['mean_load'].std():.1f}")

# ════════════════════════════════════════════════════════════════
sep("10. SIGNAL GRID VARIANCE — classification breakdown per condition")
# ════════════════════════════════════════════════════════════════
sgvc = pl.read_parquet(str(BASE / "signal_grid_variance_channel.parquet"))
print(f"Rows: {sgvc.shape[0]:,}, Columns: {sgvc.columns}")

if "variance_classification" in sgvc.columns:
    print("\n--- Global classification distribution ---")
    print(sgvc["variance_classification"].value_counts().sort("count", descending=True))
    
    print("\n--- Per-condition classification counts ---")
    print(sgvc.group_by(["condition_id", "variance_classification"]).agg(
        pl.len().alias("count")
    ).sort(["condition_id", "count"], descending=[False, True]))

# ════════════════════════════════════════════════════════════════
sep("11. KINETIC STRAIN EVENTS — what IS in there?")
# ════════════════════════════════════════════════════════════════
ks = pl.read_parquet(str(BASE / "kinetic_strain_events.parquet"))
print(f"Rows: {ks.shape[0]}, Columns: {ks.columns}")
print(ks.describe())
# Check if dt_ps actually varies
if "dt_ps" in ks.columns:
    print(f"\ndt_ps: nunique={ks['dt_ps'].n_unique()}, "
          f"min={ks['dt_ps'].min()}, max={ks['dt_ps'].max()}")

# ════════════════════════════════════════════════════════════════
sep("12. STREAM SNR MASKS — per-channel noise floor variation")
# ════════════════════════════════════════════════════════════════
snr = pl.read_parquet(str(BASE / "stream_snr_masks.parquet"))
print(f"Rows: {snr.shape[0]}, Columns: {snr.columns}")
print(snr.head(10))

# ════════════════════════════════════════════════════════════════
sep("13. TOPOLOGY — what atom/residue data is available?")
# ════════════════════════════════════════════════════════════════
for topo_file in sorted(TOPO.glob("glp1r_*.topology.json"))[:2]:
    data = json.loads(topo_file.read_text())
    print(f"\n{topo_file.name}:")
    print(f"  Top-level keys: {list(data.keys())[:20]}")
    for key in ['n_atoms', 'n_residues', 'num_atoms', 'num_residues']:
        if key in data:
            print(f"  {key}: {data[key]}")
    # Check for positions/coordinates
    for key in ['positions', 'coords', 'coordinates', 'atoms']:
        if key in data:
            val = data[key]
            if isinstance(val, list):
                print(f"  {key}: {len(val)} entries, type={type(val[0]) if val else '?'}")
            elif isinstance(val, dict):
                print(f"  {key}: dict with keys {list(val.keys())[:5]}")

# ════════════════════════════════════════════════════════════════
sep("14. INTERFACE STERIC ENVIRONMENT — unused voxel data")
# ════════════════════════════════════════════════════════════════
ise = pl.read_parquet(str(BASE / "../../track_0_manual_emulation/interface_steric_environment.parquet"))
print(f"Rows: {ise.shape[0]}, Columns: {ise.columns}")
if "variance_classification" in ise.columns:
    print(ise["variance_classification"].value_counts().sort("count", descending=True))
# Show what data each voxel carries
print(ise.head(5))

# ════════════════════════════════════════════════════════════════
sep("15. RESIDUE INDEX MAPPING — full mapping table")
# ════════════════════════════════════════════════════════════════
rim = pl.read_parquet("campaigns/glp1r_aleniglipron/topology/residue_index_mapping_matrix.parquet")
print(f"Rows: {rim.shape[0]}, Columns: {rim.columns}")
print(rim.head(10))

# ════════════════════════════════════════════════════════════════
sep("EXTRACTION COMPLETE")
# ════════════════════════════════════════════════════════════════
