"""VN-EGNN: Virtual-Node E(3)-Equivariant Graph Neural Network.

Binding site prediction via equivariant message passing augmented with
K virtual nodes whose learned positions regress to binding site centers.

Based on: Sestak et al., "VN-EGNN: E(3)-equivariant graph neural networks
with virtual nodes enhance protein binding site identification."
J. Cheminf. 2026, doi:10.1186/s13321-025-01127-9
arXiv:2404.07194, github.com/ml-jku/vnegnn

Key design choices:
  • K = 16 virtual nodes (configurable 10-20 per paper)
  • VN initialization: Fibonacci sphere at max-atom-distance radius, random rotation
  • Full bipartite connectivity VN↔protein atoms (no cutoff) — mitigates oversquashing
  • Protein–protein edges: 8Å cutoff radius
  • E(3) equivariance via EGNN-style coordinate update with invariant scalar messages
  • Hidden dim 128, 4 message-passing layers (papers: 3-4 @ 64-128)
  • Two losses: (a) atom-level binding probability (focal BCE);
                (b) VN position → nearest true ligand center (Chamfer/L2)
  • ONNX-clean: standard torch ops, index_select/scatter over fixed edge sets,
    no dynamic control flow in forward()
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def fibonacci_sphere(k: int, radius: float, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    """Quasi-uniform points on a sphere via the Fibonacci lattice.

    Returns (k, 3) positions scaled to the given radius.
    """
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    i = torch.arange(k, device=device, dtype=dtype)
    y = 1.0 - 2.0 * i / (k - 1) if k > 1 else torch.zeros_like(i)
    r_xy = torch.sqrt((1.0 - y * y).clamp(min=0.0))
    theta = phi * i
    x = torch.cos(theta) * r_xy
    z = torch.sin(theta) * r_xy
    return radius * torch.stack([x, y, z], dim=-1)


def random_rotation(device: torch.device, dtype: torch.dtype,
                    rng: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random 3x3 rotation via QR of a Gaussian matrix."""
    a = torch.randn(3, 3, generator=rng, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    d = torch.diag(torch.sign(torch.diag(r)))
    return q @ d


# ─────────────────────────────────────────────────────────────
#  Core EGNN layer (equivariant)
# ─────────────────────────────────────────────────────────────

class EGNNLayer(nn.Module):
    """Equivariant Graph Neural Network layer.

    Updates:
        m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
        x_i' = x_i + C * sum_j (x_i - x_j) * phi_x(m_ij)
        h_i' = phi_h(h_i, aggregate_j m_ij)

    Uses fixed edge_index (long tensor [2, E]) for ONNX compatibility.
    """
    def __init__(self, hidden_dim: int, edge_feat_dim: int = 0, act: str = "silu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        activation = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }[act]

        # Edge MLP: (h_i, h_j, ||dx||^2, edge_feat) → message
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feat_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
        )
        # Coord MLP: message → scalar gate on (x_i - x_j)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Node MLP: (h_i, aggregated) → new h
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Coord update coefficient (paper: C ~ 1/|N(i)|, here: learnable scalar)
        self.coord_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self,
                h: torch.Tensor,                # [N, H]
                x: torch.Tensor,                # [N, 3]
                edge_index: torch.Tensor,       # [2, E]
                edge_feat: Optional[torch.Tensor] = None,   # [E, EF]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index[0], edge_index[1]   # src (i), dst (j) — messages j→i

        h_i = h.index_select(0, row)
        h_j = h.index_select(0, col)
        dx = x.index_select(0, row) - x.index_select(0, col)
        d2 = (dx * dx).sum(dim=-1, keepdim=True)           # [E, 1]

        if edge_feat is not None:
            e_in = torch.cat([h_i, h_j, d2, edge_feat], dim=-1)
        else:
            e_in = torch.cat([h_i, h_j, d2], dim=-1)
        m_ij = self.phi_e(e_in)                            # [E, H]

        # Coord update: x_i' = x_i + scale * sum_j (x_i - x_j) * phi_x(m_ij) / |N_i|
        coord_weight = self.phi_x(m_ij)                    # [E, 1]
        coord_msg = dx * coord_weight                      # [E, 3]
        x_update = torch.zeros_like(x)
        x_update = x_update.index_add_(0, row, coord_msg)
        # Normalize by degree (avoid blow-up for high-degree nodes)
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        ones = torch.ones(row.size(0), device=x.device, dtype=x.dtype)
        deg = deg.index_add_(0, row, ones).clamp(min=1.0)
        x_update = x_update / deg.unsqueeze(-1)
        x_new = x + self.coord_scale * x_update

        # Node update: h_i' = phi_h(h_i, aggregated)
        msg_agg = torch.zeros(h.size(0), self.hidden_dim, device=h.device, dtype=h.dtype)
        msg_agg = msg_agg.index_add_(0, row, m_ij)
        h_new = h + self.phi_h(torch.cat([h, msg_agg], dim=-1))

        return h_new, x_new


# ─────────────────────────────────────────────────────────────
#  Virtual-node E(3) GNN
# ─────────────────────────────────────────────────────────────

class VNEGNN(nn.Module):
    """VN-EGNN with K virtual nodes + residue atoms on a contact graph.

    Forward inputs (all fixed-shape, ONNX-friendly):
        atom_features:  [N, F_in]          per-residue feature vector
        atom_coords:    [N, 3]              Cα coordinates
        edge_index:     [2, E]              protein–protein + bidirectional VN↔protein
        edge_feat:      [E, 1] (optional)   RBF or distance-encoded
        vn_init_coords: [K, 3]              pre-initialized virtual node positions
                                            (caller supplies — so Fibonacci+rotation
                                             is computed once per target, not in ONNX)

    Outputs:
        atom_logits:    [N, 1]              binding residue probability (pre-sigmoid)
        vn_coords:      [K, 3]              predicted binding site centers
        vn_confidence:  [K, 1]              per-VN confidence logits
    """
    def __init__(self,
                 in_dim: int = 1577,         # 25 struct + 26 NMA + 5 pert + 216 phys + 1280 ESM + 25 pad
                 hidden_dim: int = 128,
                 n_layers: int = 4,
                 n_virtual_nodes: int = 16,
                 edge_feat_dim: int = 1,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_virtual_nodes = n_virtual_nodes

        # Atom feature encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # VN embedding — a learned class embedding, 1 row per VN slot
        # (broken out per-VN to let them specialize)
        self.vn_embedding = nn.Parameter(torch.randn(n_virtual_nodes, hidden_dim) * 0.02)

        # EGNN stack
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_feat_dim=edge_feat_dim)
            for _ in range(n_layers)
        ])

        # Prediction heads
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.vn_confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                atom_features: torch.Tensor,
                atom_coords: torch.Tensor,
                edge_index: torch.Tensor,
                edge_feat: Optional[torch.Tensor] = None,
                vn_init_coords: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = atom_features.size(0)
        K = self.n_virtual_nodes

        # Encode atom features
        h_atom = self.atom_encoder(atom_features)           # [N, H]

        # Virtual node features come from the learned embedding
        h_vn = self.vn_embedding                            # [K, H]

        # Concatenate: atoms first [0..N), VNs next [N..N+K)
        h = torch.cat([h_atom, h_vn], dim=0)                # [N+K, H]

        # Initial coordinates: atoms from input, VNs from caller-supplied
        if vn_init_coords is None:
            # Fallback: zero init (caller should prefer to pass sphere init)
            vn_init_coords = torch.zeros(K, 3,
                                         device=atom_coords.device,
                                         dtype=atom_coords.dtype)
        x = torch.cat([atom_coords, vn_init_coords], dim=0)  # [N+K, 3]

        # Message passing
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_feat)

        # Heads
        atom_logits = self.atom_head(h[:N])                  # [N, 1]
        vn_coords = x[N:]                                    # [K, 3]
        vn_confidence = self.vn_confidence_head(h[N:])       # [K, 1]

        return atom_logits, vn_coords, vn_confidence


# ─────────────────────────────────────────────────────────────
#  Edge construction (not part of model forward for ONNX cleanliness)
# ─────────────────────────────────────────────────────────────

def build_edge_index(atom_coords: torch.Tensor,
                     n_virtual_nodes: int,
                     cutoff: float = 8.0,
                     ) -> torch.Tensor:
    """Build the edge_index tensor expected by VNEGNN.forward.

    Edges:
      • Protein–protein: bidirectional within `cutoff` Å (no self-loops)
      • VN–atom: fully-connected bidirectional (K × N × 2 edges)

    Returns LongTensor [2, E]. Computed once per target in Python — not
    inside the ONNX graph.
    """
    N = atom_coords.size(0)
    K = n_virtual_nodes
    device = atom_coords.device

    # Protein–protein: pairwise distances under cutoff
    d2 = torch.cdist(atom_coords, atom_coords)              # [N, N]
    mask = (d2 < cutoff) & (d2 > 1e-6)                      # no self
    pp_pairs = mask.nonzero(as_tuple=False).t()             # [2, E_pp]

    # VN↔atom: fully bipartite
    vn_ids = torch.arange(N, N + K, device=device)
    atom_ids = torch.arange(0, N, device=device)
    # Edges atom→VN: for each VN id, for each atom id
    vn_src = vn_ids.unsqueeze(1).expand(K, N).reshape(-1)   # [K*N]
    atom_dst = atom_ids.unsqueeze(0).expand(K, N).reshape(-1)
    va_edges = torch.stack([vn_src, atom_dst], dim=0)       # [2, K*N]
    av_edges = torch.stack([atom_dst, vn_src], dim=0)       # [2, K*N]

    return torch.cat([pp_pairs, va_edges, av_edges], dim=1)


def init_virtual_nodes(atom_coords: torch.Tensor,
                       n_virtual_nodes: int,
                       rng: Optional[torch.Generator] = None,
                       ) -> torch.Tensor:
    """Initialize VN coords as a Fibonacci sphere at max-atom-distance,
    then apply a random rotation to break symmetry (paper: Section 3.1).

    Returns [K, 3] positions.
    """
    center = atom_coords.mean(dim=0)
    radius = (atom_coords - center).norm(dim=-1).max().item()
    vn = fibonacci_sphere(n_virtual_nodes, radius, atom_coords.device, atom_coords.dtype)
    R = random_rotation(atom_coords.device, atom_coords.dtype, rng)
    return vn @ R.t() + center


# ─────────────────────────────────────────────────────────────
#  Losses
# ─────────────────────────────────────────────────────────────

def focal_binary_loss(logits: torch.Tensor, labels: torch.Tensor,
                      alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal BCE for imbalanced per-atom binding labels."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    p_t = p * labels + (1 - p) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return loss.mean()


def chamfer_vn_loss(pred_vn: torch.Tensor,
                    gt_centers: torch.Tensor,
                    confidence: Optional[torch.Tensor] = None,
                    ) -> torch.Tensor:
    """One-sided Chamfer: for each ground-truth center, penalize the
    distance to the CLOSEST predicted VN.

    This allows multiple VNs to collapse onto the same site without being
    punished, and lets "extra" VNs scatter harmlessly — which matches the
    paper's observation that VNs specialize to detected sites.
    """
    if gt_centers.numel() == 0:
        return torch.tensor(0.0, device=pred_vn.device, dtype=pred_vn.dtype)
    # Pairwise distances [M_gt, K_vn]
    d = torch.cdist(gt_centers.unsqueeze(0), pred_vn.unsqueeze(0)).squeeze(0)
    min_d, min_idx = d.min(dim=1)
    loss = min_d.mean()
    # If confidence provided, also push up confidence of the winning VNs
    if confidence is not None:
        winners = torch.zeros(pred_vn.size(0), device=pred_vn.device)
        winners.scatter_(0, min_idx, 1.0)
        conf_loss = F.binary_cross_entropy_with_logits(
            confidence.squeeze(-1), winners
        )
        loss = loss + 0.1 * conf_loss
    return loss


# ─────────────────────────────────────────────────────────────
#  ONNX export
# ─────────────────────────────────────────────────────────────

def export_onnx(model: VNEGNN,
                out_path: str,
                example_n_atoms: int = 300,
                opset: int = 17) -> None:
    """Export VNEGNN to ONNX with dynamic batch dimensions for inference.

    Example inputs must satisfy the actual model signature. Variable
    dimensions (number of atoms N, number of edges E) are marked as
    dynamic so the Rust ort runtime can feed arbitrary-size targets.
    """
    model.eval()
    device = next(model.parameters()).device
    in_dim = model.atom_encoder[0].in_features
    K = model.n_virtual_nodes

    # Dummy inputs for tracing
    N = example_n_atoms
    E = N * 20  # rough — 20 neighbors avg + 2*K*N bipartite; ONNX treats as dynamic
    atom_features = torch.randn(N, in_dim, device=device)
    atom_coords = torch.randn(N, 3, device=device)
    edge_index = torch.randint(0, N + K, (2, E), device=device, dtype=torch.long)
    edge_feat = torch.randn(E, 1, device=device)
    vn_init = torch.randn(K, 3, device=device)

    torch.onnx.export(
        model,
        (atom_features, atom_coords, edge_index, edge_feat, vn_init),
        out_path,
        input_names=["atom_features", "atom_coords", "edge_index", "edge_feat", "vn_init_coords"],
        output_names=["atom_logits", "vn_coords", "vn_confidence"],
        dynamic_axes={
            "atom_features": {0: "N"},
            "atom_coords": {0: "N"},
            "edge_index": {1: "E"},
            "edge_feat": {0: "E"},
            "atom_logits": {0: "N"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model → {out_path}")


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(42)
    N = 150
    K = 16
    in_dim = 1577

    atom_features = torch.randn(N, in_dim)
    atom_coords = torch.randn(N, 3) * 10.0

    model = VNEGNN(in_dim=in_dim, hidden_dim=128, n_layers=4, n_virtual_nodes=K)
    edge_index = build_edge_index(atom_coords, n_virtual_nodes=K, cutoff=8.0)
    edge_feat = torch.randn(edge_index.size(1), 1)
    vn_init = init_virtual_nodes(atom_coords, K)

    with torch.no_grad():
        a_logits, vn_xyz, vn_conf = model(atom_features, atom_coords,
                                          edge_index, edge_feat, vn_init)
    print(f"N={N}  K={K}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  atom_logits: {tuple(a_logits.shape)}")
    print(f"  vn_coords:   {tuple(vn_xyz.shape)}")
    print(f"  vn_conf:     {tuple(vn_conf.shape)}")
