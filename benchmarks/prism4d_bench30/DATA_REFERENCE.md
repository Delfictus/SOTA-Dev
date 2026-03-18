# PRISM4D-BENCH30 Data Reference

## File Layout

```
benchmarks/prism4d_bench30/
  benchmark_manifest.json          # 30 targets with metadata
  ground_truth/ligand_centroids.json  # {tid: {centroid: [x,y,z]}}
  topologies/{apo_pdb}.topology.json  # protein atom positions
  results/{tid}/{apo_pdb}.binding_sites.json  # detected sites
  site_features_all.csv            # all sites, all features, flat CSV
  benchmark_results.json           # aggregate per-target results
```

## How to load data for a target

```python
import json, os, numpy as np

# 1. Load manifest and ground truth
manifest = json.load(open('benchmarks/prism4d_bench30/benchmark_manifest.json'))
gt = json.load(open('benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json'))

# 2. For a specific target (e.g. target 8 = 1HCL):
tid = '8'
target = next(t for t in manifest['targets'] if str(t['id']) == tid)
apo_pdb = target['apo_pdb']   # '1HCL'
holo_pdb = target['holo_pdb'] # '1JSV'

# 3. Load binding sites
sites_dir = f'benchmarks/prism4d_bench30/results/{tid}'
bs_file = next(f for f in os.listdir(sites_dir) if f.endswith('.binding_sites.json'))
data = json.load(open(os.path.join(sites_dir, bs_file)))
sites = data['sites']  # list of site dicts

# 4. Load protein atom positions
topo = json.load(open(f'benchmarks/prism4d_bench30/topologies/{apo_pdb.lower()}.topology.json'))
positions = np.array(topo['positions'], dtype=np.float32).reshape(-1, 3)

# 5. Get ligand centroid (ground truth)
lig_centroid = np.array(gt[tid]['centroid'])

# 6. Compute DCC for each site (NOT stored in JSON — must compute)
for site in sites:
    centroid = np.array(site['centroid'])
    dcc = float(np.linalg.norm(centroid - lig_centroid))
```

## Manifest target fields

```python
{
    "id": 8,                    # target number (use as string for dict keys)
    "apo_pdb": "1HCL",         # apo structure PDB code
    "holo_pdb": "1JSV",        # holo structure PDB code
    "ligand_resname": "U55",   # ligand residue name in holo
    "site_type": "orthosteric", # orthosteric|cryptic|allosteric|PPI
    "topology_file": "topologies/1hcl.topology.json"
}
```

## APO PDB to target ID mapping

| TID | APO  | HOLO | LIG | Type        |
|-----|------|------|-----|-------------|
| 1   | 1JWP | 1PZO | CBT | cryptic     |
| 2   | 2NPQ | 2ZB1 | GK4 | cryptic     |
| 3   | 1KV1 | 1KV2 | B96 | allosteric  |
| 4   | 1MY0 | 1N0T | AT1 | cryptic     |
| 5   | 4EY4 | 4EY7 | E20 | orthosteric |
| 6   | 2HNP | 1T49 | 892 | allosteric  |
| 7   | 1M47 | 1PY2 | FRH | PPI         |
| 8   | 1HCL | 1JSV | U55 | orthosteric |
| 9   | 1YES | 1YET | GDM | orthosteric |
| 10  | 2OSS | 3MXF | JQ1 | orthosteric |
| 11  | 1FKG | 1FKJ | FK5 | orthosteric |
| 12  | 1HPV | 1HVR | XK2 | orthosteric |
| 13  | 3ERT | 1ERR | RAL | orthosteric |
| 14  | 1K3F | 1K3G | HEC | orthosteric |
| 15  | 1P38 | 3HEC | STI | cryptic     |
| 16  | 1YVF | 1YV3 | ADP | allosteric  |
| 17  | 1PKL | 1PKN | PYR | orthosteric |
| 18  | 1DWD | 1DWC | MIT | orthosteric |
| 19  | 1TTH | 1TTI | PGA | orthosteric |
| 20  | 2CBA | 3HS4 | AZM | orthosteric |
| 21  | 1RDQ | 1K1J | FD2 | orthosteric |
| 22  | 3L3N | 3L3M | A92 | orthosteric |
| 23  | 1NNA | 1NNC | ZMR | orthosteric |
| 24  | 1TPA | 1TPP | APA | orthosteric |
| 25  | 1ABF | 1ABE | ARA | orthosteric |
| 26  | 4HT0 | 4HT2 | V50 | orthosteric |
| 27  | 1C5Y | 1C5X | FLC | orthosteric |
| 28  | 1GKC | 1GKD | STN | orthosteric |
| 29  | 3TMN | 5TMN | 0PJ | orthosteric |
| 30  | 4DFR | 3DFR | MTX | orthosteric |

## Site JSON fields (per entry in sites[])

**CRITICAL: There is NO `site['x']`, `site['y']`, `site['z']`.
Centroid is `site['centroid']` → `[x, y, z]` list.**

**CRITICAL: There is NO `site['dcc']`.
DCC must be computed: `np.linalg.norm(np.array(site['centroid']) - lig_centroid)`**

```
centroid              [float, float, float]  # [x, y, z] in Angstroms
id                    int                    # cluster ID
quality_score         float                  # composite ranking score
volume                float                  # pocket volume (A^3)
spike_count           int                    # total spikes in pocket
engine_geo            float                  # Cobb-Douglas geometry engine score
engine_chem           float                  # Cobb-Douglas chemistry engine score
engine_phys           float                  # Cobb-Douglas physics engine score
engine_vcs            float                  # Cobb-Douglas VCS engine score
burial_score          float                  # sigmoid(mean_residues, center=3)
mean_burial           float                  # mean n_residues of local spikes
onset_score           float                  # 1 - (median_ts / max_ts)
source_diversity      float                  # UV/LIF balance + EFP bonus
aromatic_score        float                  # aromatic proximity
catalytic_residue_count int                  # count of known catalytic residues
druggability          float                  # druggability composite
is_druggable          bool                   # druggability threshold
sphericity            float                  # eigenvalue ratio (0=elongated, 1=sphere)
wd_coherence          float                  # variance of wd_change values
breathing_score       float                  # CV of per-frame burial
frustrated_solvent_score float               # early-onset water displacement proxy
asymmetry_offset      float                  # |CoM_spikes - centroid| distance
ray_escape_ratio      float                  # Dmax/Dmin of spike projections
ccns_tau              float                  # CCNS criticality exponent
hysteresis_asymmetry  float                  # thermal hysteresis score
relative_asymmetry    float                  # relative thermal asymmetry
tide_coupling_score   float                  # overlap of TIDE triggers with lining
therm_class           str                    # "RESPONSIVE" / "CRYPTIC" / etc
classification        str                    # "ActiveSite" / "Cryptic" / etc
lining_residues       [dict, ...]            # list of {resid, resname, chain, ...}
residue_ids           [int, ...]             # residue IDs of lining
tide_trigger_residues [int, ...]             # top-5 TIDE trigger residue IDs
```

## Topology JSON fields

```
source_pdb      str              # PDB code
n_atoms         int              # total atoms
n_residues      int              # total residues
positions       [float, ...]     # flat array, reshape to (n_atoms, 3)
masses          [float, ...]     # per-atom masses
elements        [str, ...]       # per-atom element symbols
atom_names      [str, ...]       # per-atom IUPAC names
residue_names   [str, ...]       # per-atom residue names
residue_ids     [int, ...]       # per-atom residue IDs
```

## Common mistakes

1. `site['x']` → **WRONG.** Use `site['centroid'][0]`
2. `site['dcc']` → **WRONG.** Compute it from centroid vs ground truth.
3. `results/1/1hcl...` → **WRONG.** 1HCL is target 8, path is `results/8/`
4. `top['positions']` is flat → **MUST** reshape: `np.array(top['positions']).reshape(-1, 3)`
5. `python script.py` → **WRONG.** Use `python3 script.py`
