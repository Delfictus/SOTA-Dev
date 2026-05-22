# STRUCTURAL VISUALIZATION RUNBOOK — BLIND VALIDATION
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Post-freeze only

All structural overlays involving holo coordinates are post-freeze only. Pre-freeze: only apo structure visualization is permitted.

---

## Per-target visualization (post-freeze)

The validator generates `pymol_overlay.pml` automatically. Load in PyMOL:
```bash
pymol /mnt/storage/prism-outputs/blind_validation/post_freeze_validation/${TARGET}/pymol_overlay.pml
```

---

## Publication-quality figure generation

Use the existing renderer from the pub run package:
```bash
cd /home/diddy/Downloads/PRISM4D_figures_and_pymol_package/pymol/

# Create a blind validation config based on template
cp prism4d_targets_config_template.json blind_val_targets_config.json
# Then populate with B01–B09 target data (not B10 hard negative)

pymol -cq render_prism4d_panels.py -- blind_val_targets_config.json
```

---

## Config fields to populate per blind target

After post-freeze validation, extract from kcc_visualization.json:
```bash
python3 - << 'EOF'
import json, sys

TARGET = "HRAS_Q61H"
KCC_FILE = f"/mnt/storage/prism-outputs/blind_validation/B01_{TARGET}/run/*.kcc_visualization.json"

import glob
kcc_path = glob.glob(KCC_FILE)[0]
with open(kcc_path) as f:
    kcc = json.load(f)

print("centroids:", [(s["id"], s["centroid"]) for s in kcc.get("sites", [])])
print("n_sites:", kcc.get("binding_sites", 0))
for s in kcc.get("sites", []):
    print(f"  site {s['id']}: candidates = {s['kcc']['candidate_residue_ids'][:5]}...")
    print(f"  site {s['id']}: driver = {s['kcc']['driver_residue_id']}")
EOF
```

---

## Residue ID translation

Topology residue IDs → PDB author residue numbers:
```bash
python3 scripts/prism-lookup-residue.py \
    --topology ${BLIND_BASE}/${TARGET}/topologies/*.topology.json \
    --residue-id <topology_id>
```

Or use offset formula: `pdb_resnum = topology_resnum + (first_pdb_resnum - 1)`

---

## PyMOL session per target (engine-native)

The engine generates `<target>.kcc_session.pml` and `<target>.binding_sites.pml` in the run dir. Load directly:
```bash
pymol ${BLIND_BASE}/${TARGET}/run/*.kcc_session.pml
```

---

## ChimeraX

Engine-native: `<target>.binding_sites.cxc`. Open with:
```bash
ChimeraX ${BLIND_BASE}/${TARGET}/run/*.binding_sites.cxc
```

---

## Figure panels for blind validation report

For each target B01–B09:
1. Apo overview (cartoon, PRISM manifold as surface)
2. Site close-up (candidate residues as sticks, centroid sphere)
3. Holo overlay (aligned holo ligand + PRISM manifold + ligand shell at 8 Å)
4. Causal driver close-up (driver residue + candidate residues)

For B10 ADRB2 (hard negative):
1. Apo overview only + detected sites
2. Caption: "No cryptic pocket detected at orthosteric site (hard negative pass)"
