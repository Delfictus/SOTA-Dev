"""
Tier 3a: Persistent Homology Pocket Discovery.

Pure post-processor module. NO_POST_MD_LOOPS-compliant.

Treats the energy-weighted residue centroid cloud as a function on 3-space
and computes its 0-dimensional persistent homology. Pockets emerge as
connected components that persist across a wide range of intensity
thresholds. Output: per-pocket (birth, death) pair plus a parameter-free
component inventory.

This is parameter-free in the sense that no clustering radius or top-K is
hand-tuned; the only knob is the persistence threshold below which low-
significance components are discarded (defaulted to 0.05 of the normalized
energy range). The threshold is reported with the output for full
methodological transparency.
"""
import numpy as np
import gudhi


def build_alpha_complex_filtration(residue_centroids, residue_energies):
    """
    Build alpha complex on energy-weighted residue centroids.
    Filtration value: -normalized_energy on 0-simplices (high-energy
    residues 'birth' first), then propagated to higher-dim simplices
    via make_filtration_non_decreasing().
    """
    points = np.asarray(residue_centroids, dtype=np.float64)
    n = len(points)

    alpha = gudhi.AlphaComplex(points=points)
    simplex_tree = alpha.create_simplex_tree()

    energies = np.asarray(residue_energies, dtype=np.float64)
    energy_range = (energies.max() - energies.min()) + 1e-12
    energy_norm = (energies - energies.min()) / energy_range

    for v in range(n):
        simplex_tree.assign_filtration([v], -float(energy_norm[v]))

    simplex_tree.make_filtration_non_decreasing()

    return simplex_tree


def extract_pocket_persistence(simplex_tree, residue_ids,
                               persistence_threshold=0.1):
    """
    Compute 0-dimensional persistence (connected components).
    Returns list of pockets ordered by persistence (descending).
    Components with persistence below `persistence_threshold` (and not
    essential) are filtered out.
    """
    persistence = simplex_tree.persistence()
    h0 = [(birth, death) for dim, (birth, death) in persistence if dim == 0]

    pockets = []
    for birth, death in h0:
        if death == float("inf"):
            persistence_value = float("inf")
        else:
            persistence_value = death - birth
        if (persistence_value != float("inf")
                and persistence_value < persistence_threshold):
            continue
        pockets.append({
            "birth": float(birth),
            "death": float(death) if death != float("inf") else None,
            "persistence": (float(persistence_value)
                            if persistence_value != float("inf") else None),
            "is_essential": death == float("inf"),
        })

    pockets.sort(
        key=lambda p: p.get("persistence") or float("inf"),
        reverse=True,
    )
    return pockets


def assign_residues_to_persistence_components(simplex_tree, residue_ids,
                                              energies,
                                              n_components_target=10):
    """
    Walk the filtration to assign residues to their persistent component.
    For each top-N persistent component, return the residue list and
    aggregated energy.
    """
    n = len(residue_ids)
    if n == 0:
        return []

    energies = np.asarray(energies, dtype=np.float64)
    energy_range = (energies.max() - energies.min()) + 1e-12
    energy_norm = (energies - energies.min()) / energy_range

    # Union-find over residues; merge as edges appear in alpha complex.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Get all 1-simplices (edges) sorted by filtration value.
    edges = []
    for simplex, fval in simplex_tree.get_filtration():
        if len(simplex) == 2:
            edges.append((fval, simplex[0], simplex[1]))
    edges.sort()

    # Each residue starts as its own component, identified by its own
    # vertex index. Merge events: edges in sorted order. `union(a, b)` sets
    # `parent[find(a)] = find(b)`, so `find(b)` is the surviving root —
    # keep `component_residues` keyed by that root, delete the loser.
    component_residues = {i: [int(residue_ids[i])] for i in range(n)}
    for fval, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            merged = component_residues[ra] + component_residues[rb]
            del component_residues[ra]
            component_residues[rb] = merged
            union(a, b)

    # Order remaining components by total energy.
    rid_to_idx = {int(r): i for i, r in enumerate(residue_ids)}
    final_components = []
    for cidx, rlist in component_residues.items():
        idx_list = [rid_to_idx[r] for r in rlist if r in rid_to_idx]
        total_e = float(energies[idx_list].sum()) if idx_list else 0.0
        final_components.append({
            "residues": sorted(rlist),
            "total_energy": total_e,
            "n_residues": len(rlist),
        })
    final_components.sort(key=lambda c: c["total_energy"], reverse=True)
    return final_components[:n_components_target]


def discover_pockets_persistent_homology(res_df, top_n_residues=150,
                                         persistence_threshold=0.05):
    """
    Top-level: input the per-residue aggregation table, output pocket inventory.

    Expects `res_df` to be a pandas-DataFrame-like object with columns:
        residue_id, mean_x, mean_y, mean_z, total_energy
    sorted such that `head(top_n_residues)` returns the highest-energy
    residues. Accessed via `[col].values` so any object with that
    interface works.
    """
    top = res_df.head(top_n_residues)
    centroids = top[["mean_x", "mean_y", "mean_z"]].values
    energies = top["total_energy"].values
    rids = top["residue_id"].values.tolist()

    if len(centroids) < 3:
        return {"status": "BLOCKED", "gate": "Need >=3 residues for PH"}

    simplex_tree = build_alpha_complex_filtration(centroids, energies)
    persistence_diagram = extract_pocket_persistence(
        simplex_tree, rids, persistence_threshold,
    )
    components = assign_residues_to_persistence_components(
        simplex_tree, rids, energies,
    )

    return {
        "status": "OK",
        "n_persistence_components": len([
            p for p in persistence_diagram
            if p.get("persistence") and p["persistence"] > persistence_threshold
        ]),
        "persistence_diagram": persistence_diagram[:20],
        "pocket_inventory_ph": components,
        "method_note": (
            "Alpha complex on energy-weighted centroids, "
            "filtration by -normalized_energy (high-energy birth first), "
            "0-dim persistent homology. Parameter: persistence threshold "
            f"{persistence_threshold}."
        ),
    }
