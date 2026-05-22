#!/usr/bin/env python3
"""
Extended structural + NMA feature extractor v2 (least-naive).

Adds beyond v002's 26+26+5=57 dims:
  - DSSP-8 secondary structure (5 extra dims over DSSP-3)
  - Per-AA chemistry vocabulary (charge, polar, aromatic, hydrophobicity_kd)
  - Per-AA volume + flexibility prior (literature)
  - Per-residue Cα and sidechain centroid coordinates (6 dims)
  - Relative SASA (RSA = SASA / max_SASA per AA)
  - Heavy-atom contact density at 6/8/12 Å radii
  - GNM (Gaussian Network Model) lowest modes — independent of ProDy ANM
  - Per-residue cross-correlation centrality from NMA modes
  - Per-residue mode entropy
  - Per-residue normal-mode participation ratio

Total: ~95 dims per residue (vs 57 in v002).

Outputs npz keys:
  structural        (26)   — same as v002 (AA one-hot + hydrophobicity + DSSP-3 + SASA + B-factor)
  nma               (26)   — same as v002 (ProDy ANM)
  perturbed_nma     (5)    — same as v002 (derived perturbations)
  structural_ext    (~25)  — NEW: DSSP-8 delta, chemistry, geometry, contact density, RSA
  nma_ext           (~15)  — NEW: GNM modes, cross-corr centrality, mode entropy, participation ratio
  ca_xyz            (3)    — Cα coordinates per residue
  sidechain_xyz     (3)    — Sidechain centroid coordinates per residue
  resname           int8 array — AA index for ordering
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "prism-ai-inference"))

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','HID':'H','HIE':'H','HIP':'H','ILE':'I','LEU':'L',
    'LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V','CYX':'C','MSE':'M','SEP':'S','TPO':'T','PTR':'Y',
}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA_ORDER)}

CHARGE = {'D':-1,'E':-1,'K':+1,'R':+1,'H':0.5}
POLAR = set("NQSTYDEKRHC")
AROMATIC = set("FYWH")
HYDROPHOBICITY_KD = {  # Kyte-Doolittle
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
    'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
}
VOLUME = {  # Å³ from Zamyatnin 1972
    'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
    'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
    'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0,
}
MAX_SASA = {  # Tien et al. 2013 Theoretical maxima
    'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,
    'H':224,'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,
    'T':172,'W':285,'Y':263,'V':174,
}
FLEXIBILITY = {  # Vihinen et al. — average flexibility per residue
    'A':0.357,'R':0.529,'N':0.463,'D':0.511,'C':0.346,'Q':0.493,'E':0.497,
    'G':0.544,'H':0.323,'I':0.462,'L':0.365,'K':0.466,'M':0.295,'F':0.314,
    'P':0.509,'S':0.507,'T':0.444,'W':0.305,'Y':0.420,'V':0.386,
}

DSSP8_ORDER = "HEGITSCB"
DSSP8_IDX = {ss: i for i, ss in enumerate(DSSP8_ORDER)}


def extract_v2(pdb_path: Path, chain: Optional[str] = None):
    from predict import parse_pdb, extract_structural_features, extract_nma_features

    parsed = parse_pdb(str(pdb_path), chain=chain)
    n_res = parsed["n_residues"]
    if n_res == 0:
        return None

    structural = extract_structural_features(parsed).astype(np.float32)
    nma = extract_nma_features(parsed).astype(np.float32)

    structural_ext, ca_xyz, sidechain_xyz = compute_structural_ext(parsed, pdb_path)
    nma_ext = compute_nma_ext(parsed, pdb_path, nma)
    perturbed_nma = compute_perturbed_nma(nma)
    resname = build_resname_array(parsed)

    return {
        "structural": structural,
        "nma": nma,
        "perturbed_nma": perturbed_nma.astype(np.float32),
        "structural_ext": structural_ext.astype(np.float32),
        "nma_ext": nma_ext.astype(np.float32),
        "ca_xyz": ca_xyz.astype(np.float32),
        "sidechain_xyz": sidechain_xyz.astype(np.float32),
        "resname": resname.astype(np.int8),
        "n_residues": np.int32(n_res),
    }


def build_resname_array(parsed) -> np.ndarray:
    n = parsed["n_residues"]
    arr = np.full(n, -1, dtype=np.int8)
    for i, r in enumerate(parsed["residues"]):
        aa1 = AA3TO1.get(r["resname"], "X")
        arr[i] = AA_IDX.get(aa1, -1)
    return arr


def compute_structural_ext(parsed, pdb_path: Path):
    n = parsed["n_residues"]
    DIM = 25
    out = np.zeros((n, DIM), dtype=np.float32)
    ca = np.zeros((n, 3), dtype=np.float32)
    sc = np.zeros((n, 3), dtype=np.float32)

    for i, r in enumerate(parsed["residues"]):
        aa1 = AA3TO1.get(r["resname"], "X")
        out[i, 0] = CHARGE.get(aa1, 0.0)
        out[i, 1] = 1.0 if aa1 in POLAR else 0.0
        out[i, 2] = 1.0 if aa1 in AROMATIC else 0.0
        out[i, 3] = HYDROPHOBICITY_KD.get(aa1, 0.0) / 5.0
        out[i, 4] = VOLUME.get(aa1, 100.0) / 250.0
        out[i, 5] = FLEXIBILITY.get(aa1, 0.4)
        out[i, 6] = MAX_SASA.get(aa1, 200) / 300.0
        ca_xyz = r.get("ca_xyz")
        if ca_xyz is not None:
            ca[i] = ca_xyz

    try:
        import mdtraj
        t = mdtraj.load(str(pdb_path))
        sel = t.topology.select("protein")
        if len(sel) > 0:
            t = t.atom_slice(sel)
        dssp8 = mdtraj.compute_dssp(t, simplified=False)[0]
        for i, ss in enumerate(dssp8[:n]):
            j = DSSP8_IDX.get(ss, 7)
            out[i, 7 + j] = 1.0
    except Exception:
        pass

    try:
        import mdtraj
        t = mdtraj.load(str(pdb_path))
        sel = t.topology.select("protein")
        if len(sel) > 0:
            t = t.atom_slice(sel)
        sasa = mdtraj.shrake_rupley(t, mode="residue")[0]
        for i, sa in enumerate(sasa[:n]):
            aa1 = AA3TO1.get(parsed["residues"][i]["resname"], "X")
            max_sasa = MAX_SASA.get(aa1, 200)
            out[i, 15] = float(sa) / max(max_sasa, 1.0)
    except Exception:
        pass

    try:
        import mdtraj
        t = mdtraj.load(str(pdb_path))
        protein_idx = t.topology.select("protein")
        if len(protein_idx) > 0:
            t = t.atom_slice(protein_idx)
        ca_idx = t.topology.select("name CA and protein")
        if len(ca_idx) >= n:
            ca_coords = t.xyz[0, ca_idx[:n]] * 10.0
            ca[:] = ca_coords
        for r_idx, residue in enumerate(t.topology.residues):
            if r_idx >= n:
                break
            sc_atoms = [a.index for a in residue.atoms if a.name not in ("N", "CA", "C", "O", "H", "HA")]
            if sc_atoms:
                sc[r_idx] = t.xyz[0, sc_atoms].mean(axis=0) * 10.0
            else:
                sc[r_idx] = ca[r_idx]

        atoms = t.xyz[0] * 10.0
        from scipy.spatial import cKDTree
        tree = cKDTree(atoms)
        ca_pos = ca / 10.0 * 10.0
        for i in range(n):
            for j, radius in enumerate((6.0, 8.0, 12.0)):
                neigh = tree.query_ball_point(ca[i], radius)
                out[i, 16 + j] = len(neigh) / 100.0

        coords = ca
        for i in range(n):
            lo = max(0, i - 3)
            hi = min(n, i + 4)
            if hi - lo > 1:
                local = coords[lo:hi]
                centroid = local.mean(axis=0)
                rg = float(np.sqrt(((local - centroid) ** 2).sum(axis=1).mean()))
                out[i, 19] = rg / 10.0
    except Exception:
        pass

    return out, ca, sc


def compute_nma_ext(parsed, pdb_path: Path, nma: np.ndarray) -> np.ndarray:
    n = parsed["n_residues"]
    DIM = 15
    out = np.zeros((n, DIM), dtype=np.float32)

    try:
        import prody
        prody.confProDy(verbosity="none")
        protein = prody.parsePDB(str(pdb_path))
        calphas = protein.select("calpha")
        if calphas is None or calphas.numAtoms() == 0:
            return out

        gnm = prody.GNM("gnm")
        gnm.buildKirchhoff(calphas, cutoff=10.0)
        gnm.calcModes(n_modes=10)
        gnm_modes = gnm.getEigvecs()
        n_calphas = gnm_modes.shape[0]
        n_use = min(n, n_calphas)
        for k in range(min(5, gnm_modes.shape[1])):
            out[:n_use, k] = gnm_modes[:n_use, k]

        sqf = prody.calcSqFlucts(gnm)
        out[:n_use, 5] = sqf[:n_use] / max(float(sqf.max()), 1e-6)

        cross_corr = prody.calcCrossCorr(gnm)
        if cross_corr.shape[0] >= n_use:
            out[:n_use, 6] = cross_corr[:n_use].mean(axis=1)
            out[:n_use, 7] = np.abs(cross_corr[:n_use]).mean(axis=1)
    except Exception:
        pass

    if nma.shape[1] >= 20:
        modes = nma[:, :20]
        p = np.abs(modes)
        p_sum = p.sum(axis=1, keepdims=True)
        p_norm = p / np.maximum(p_sum, 1e-12)
        entropy = -np.where(p_norm > 0, p_norm * np.log(np.maximum(p_norm, 1e-12)), 0.0).sum(axis=1)
        out[:, 8] = entropy / np.log(20)

        participation = (p_sum ** 2).flatten() / np.maximum((modes ** 2).sum(axis=1) * 20, 1e-12)
        out[:, 9] = participation

        for k in range(min(5, modes.shape[1])):
            mode = modes[:, k]
            for i in range(n):
                lo, hi = max(0, i - 4), min(n, i + 5)
                if hi - lo > 1:
                    local_var = float(np.var(mode[lo:hi]))
                    out[i, 10 + k] = local_var

    return out


def compute_perturbed_nma(nma: np.ndarray) -> np.ndarray:
    n_res, n_dim = nma.shape
    if n_dim < 20:
        return np.zeros((n_res, 5), dtype=np.float32)
    mode_disps = nma[:, :20]
    mean_disp = mode_disps.mean(axis=1)
    max_disp = mode_disps.max(axis=1)
    low_freq = mode_disps[:, :5].mean(axis=1)
    high_freq = mode_disps[:, 15:20].mean(axis=1)
    smoothed = np.zeros(n_res, dtype=np.float32)
    for i in range(n_res):
        lo, hi = max(0, i - 2), min(n_res, i + 3)
        smoothed[i] = mode_disps[lo:hi].mean()
    grad = np.zeros(n_res, dtype=np.float32)
    grad[:-1] = smoothed[1:] - smoothed[:-1]
    return np.stack([mean_disp, max_disp, low_freq, high_freq, grad], axis=1)


def find_pdb(arrow_dir: Path, base: str) -> Optional[Path]:
    parent = arrow_dir.parent
    grandparent = parent.parent
    candidates = [
        parent / "2_clean" / f"{base}.pdb",
        parent / "2_clean" / f"{base}_clean.pdb",
        parent / "prep" / f"{base}_clean.pdb",
        parent / "prep" / f"{base}.pdb",
        grandparent / "2_clean" / f"{base}.pdb",
        grandparent / "prep" / f"{base}_clean.pdb",
        arrow_dir / f"{base}_clean.pdb",
        arrow_dir / f"{base}.pdb",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", type=Path, default=None)
    ap.add_argument("--targets-from-arrow-dirs", type=Path, action="append", default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--chain", type=str, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb:
        result = extract_v2(args.pdb, chain=args.chain)
        if result:
            out = args.output_dir / f"{args.pdb.stem}_structural_v2.npz"
            np.savez_compressed(out, **result)
            print(f"wrote {out}: structural={result['structural'].shape} nma={result['nma'].shape} ext={result['structural_ext'].shape}/{result['nma_ext'].shape}")
        return

    if not args.targets_from_arrow_dirs:
        print("need --pdb or --targets-from-arrow-dirs", file=sys.stderr)
        return

    ok, fail = 0, 0
    for root in args.targets_from_arrow_dirs:
        for arrow in root.rglob("*.topology.spike_events.arrow"):
            if arrow.stat().st_size < 1_000_000_000:
                continue
            base = arrow.name.replace(".topology.spike_events.arrow", "")
            pdb = find_pdb(arrow.parent, base)
            if not pdb:
                print(f"[skip] no PDB for {base}")
                fail += 1
                continue
            out = args.output_dir / f"{base}_structural_v2.npz"
            if out.exists():
                print(f"[skip] {base} (exists)")
                ok += 1
                continue
            print(f"[run] {base} ← {pdb}")
            try:
                result = extract_v2(pdb, chain=args.chain)
                if result is None:
                    print(f"[fail] {base}: 0 residues")
                    fail += 1
                    continue
                np.savez_compressed(out, **result)
                print(f"  → {out.stat().st_size/1024:.1f} KB  shapes: struct={result['structural'].shape} ext={result['structural_ext'].shape} nma={result['nma'].shape} nma_ext={result['nma_ext'].shape}")
                ok += 1
            except Exception as e:
                print(f"[fail] {base}: {e}")
                fail += 1
    print(f"\n=== done: {ok} ok, {fail} fail ===")


if __name__ == "__main__":
    main()
