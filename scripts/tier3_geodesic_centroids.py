"""
Tier 3b: Geodesic Fréchet Centroids on Protein Backbone Manifold.

Pure post-processor module. NO_POST_MD_LOOPS-compliant.

Replaces Euclidean energy-weighted residue centroid computation with
geodesic centroids constrained to the backbone CA chain. Produces centroid
positions that ALWAYS lie on the protein structure, eliminating the §4
megacluster-collapse failure mode by construction.

Method: backbone CA chain forms a path graph; geodesic distance between
residues = sum of CA-CA edge lengths along the path. Fréchet mean of a
set of residues = backbone position minimizing sum of squared geodesic
distances. For residues on a single contiguous chain, this is the median
residue position weighted by energy.

Side-effect output: per-pocket `euclidean_to_geodesic_delta_A` flags
megacluster-collapse cases (delta > 5Å) by direct comparison against the
naive Euclidean energy-weighted centroid.
"""
import numpy as np
import networkx as nx


def build_backbone_graph(md_atom_positions):
    """
    Build NetworkX graph: nodes = residue IDs (with CA position as
    attribute), edges = sequence-adjacent residues weighted by Euclidean
    CA-CA distance.

    `md_atom_positions` is the dict produced by
    `tier1_anchor_hellinger.load_topology_atom_positions`: residue_id ->
    np.array of all heavy-atom coords for that residue. We approximate
    each residue's CA position as the mean of its atoms (the actual CA
    is included; mean-of-heavy-atoms is the conventional residue locus
    when the explicit CA atom name isn't tagged).
    """
    G = nx.Graph()
    sorted_residues = sorted(md_atom_positions.keys())

    ca_per_res = {
        rid: md_atom_positions[rid].mean(axis=0)
        for rid in sorted_residues
    }

    for rid in sorted_residues:
        G.add_node(rid, ca=ca_per_res[rid])

    # Sequential edges. Sequence-adjacent residues (resseq diff == 1)
    # get the literal CA-CA distance as the edge weight. Sequence breaks
    # (gaps) get a 10x penalty so geodesic paths avoid crossing them
    # whenever possible.
    for i in range(len(sorted_residues) - 1):
        a, b = sorted_residues[i], sorted_residues[i + 1]
        if b - a == 1:
            d = float(np.linalg.norm(ca_per_res[a] - ca_per_res[b]))
            G.add_edge(a, b, weight=d)
        else:
            d_eucl = float(np.linalg.norm(ca_per_res[a] - ca_per_res[b]))
            G.add_edge(a, b, weight=d_eucl * 10.0)

    return G, ca_per_res


def geodesic_frechet_centroid(pocket_residues, residue_energies,
                              backbone_graph, ca_per_res):
    """
    Compute geodesic Fréchet mean: residue minimizing sum of squared
    geodesic distances weighted by energy. Returns
    `(best_residue_id, [x, y, z], spread_A)` or `(None, None, None)` if
    no candidate has a valid path-length sum.
    """
    candidates = [r for r in pocket_residues if r in backbone_graph.nodes]
    if not candidates:
        return None, None, None

    weights = np.array([residue_energies.get(r, 0.0) for r in candidates])
    if weights.sum() == 0:
        return None, None, None
    weights = weights / weights.sum()

    best_r = None
    best_loss = float("inf")
    for r_candidate in candidates:
        try:
            loss = 0.0
            for r_target, w in zip(candidates, weights):
                if r_candidate == r_target:
                    continue
                d = nx.shortest_path_length(
                    backbone_graph, r_candidate, r_target, weight="weight",
                )
                loss += w * (d ** 2)
            if loss < best_loss:
                best_loss = loss
                best_r = r_candidate
        except nx.NetworkXNoPath:
            continue

    if best_r is None:
        return None, None, None

    return (best_r,
            ca_per_res[best_r].tolist(),
            float(np.sqrt(best_loss)))


def compute_geodesic_centroids_for_pockets(pockets, residue_energies_per_pocket,
                                            md_atom_positions):
    """
    Top-level: for each pocket, compute geodesic Fréchet centroid AND
    its delta against the naive Euclidean energy-weighted centroid.

    `pockets`                       : dict {pocket_id: [residue_ids]}
    `residue_energies_per_pocket`   : dict {pocket_id: {residue_id: energy}}
    `md_atom_positions`             : dict {residue_id: np.array(N_atoms, 3)}

    Returns dict {pocket_id: { ... , "interpretation": <enum> }}.
    The `interpretation` flags megacluster-collapse cases:
      GEODESIC_AGREES_EUCLIDEAN  (delta < 2.0 Å)
      MILD_DIVERGENCE            (delta < 5.0 Å)
      MEGACLUSTER_COLLAPSE_DETECTED  (delta >= 5.0 Å — §4 failure mode)
    """
    if not md_atom_positions:
        return {"status": "BLOCKED", "gate": "Need MD atom positions"}

    G, ca_per_res = build_backbone_graph(md_atom_positions)

    out = {}
    for pidx, pocket_residues in pockets.items():
        energies = residue_energies_per_pocket.get(pidx, {})
        anchor_r, anchor_xyz, spread = geodesic_frechet_centroid(
            pocket_residues, energies, G, ca_per_res,
        )
        if anchor_r is None:
            out[pidx] = {
                "status": "BLOCKED",
                "gate": "No backbone-graph residues in pocket",
            }
            continue

        # Compare to Euclidean centroid for §4 megacluster diagnostic.
        coords = np.array([
            ca_per_res[r] for r in pocket_residues if r in ca_per_res
        ])
        if len(coords) > 0:
            energy_array = np.array([
                energies.get(r, 0.0)
                for r in pocket_residues if r in ca_per_res
            ])
            if energy_array.sum() > 0:
                eucl_centroid = np.average(coords, axis=0, weights=energy_array)
            else:
                eucl_centroid = coords.mean(axis=0)
            megacluster_delta = float(
                np.linalg.norm(np.asarray(anchor_xyz) - eucl_centroid)
            )
        else:
            megacluster_delta = float("nan")

        out[pidx] = {
            "status": "OK",
            "geodesic_anchor_residue": int(anchor_r),
            "geodesic_anchor_xyz": anchor_xyz,
            "geodesic_spread_A": spread,
            "euclidean_to_geodesic_delta_A": megacluster_delta,
            "interpretation": (
                "GEODESIC_AGREES_EUCLIDEAN" if megacluster_delta < 2.0 else
                "MILD_DIVERGENCE" if megacluster_delta < 5.0 else
                "MEGACLUSTER_COLLAPSE_DETECTED"  # §4 case
            ),
        }
    return out
