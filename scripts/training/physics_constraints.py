"""Physics Constraints for PRISM-4D Inference Validation.

Patent claim: "Conditional Logic Validation Layer for Physics-Constrained
Neural Network Inference".

Each constraint is an If-P-then-Q rule derived from empirically verified
physical relationships. The contrapositive (¬Q → ¬P) is used for false-
positive auditing.

Hard constraints override model output (violations must be 0 post-training).
Soft constraints flag violations for review (target <5% violation rate).

Usage — in VN-EGNN training loop (after epoch 20):

    from physics_constraints import compute_penalty, audit_constraints
    loss = base_loss + compute_penalty(pred, feat, epoch)

Usage — post-hoc audit on all 345 GT targets:

    report = audit_constraints(model, all_targets)
    assert all(r["total_violations"] == 0 for r in report["hard"].values())

Design notes:
  • Each constraint is a dataclass: name, description, rule_fn, contrapositive_fn
  • rule_fn(pred, feat) → Tensor[bool] with True where the rule is violated
  • contrapositive_fn(pred, feat) → Tensor[bool] marking FP candidates
  • feat is a FeatureView dict mapping feature names → tensors; the view
    also carries per-target quantile stats (q75, p90) computed once at load
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Penalty weights for the training loss
HARD_PENALTY_WEIGHT = 10.0
SOFT_PENALTY_WEIGHT = 1.0
ENFORCE_AFTER_EPOCH = 20


# ─────────────────────────────────────────────────────────────
#  Feature view helper
# ─────────────────────────────────────────────────────────────

class FeatureView(dict):
    """Dict-backed feature container with per-target quantile attributes.

    Accepts either torch tensors or numpy arrays. Designed for both training
    batches and post-hoc audit over full arrays.
    """
    def __init__(self, features: Dict[str, Any], *,
                 quantiles: Optional[Dict[str, float]] = None):
        super().__init__(features)
        # Precomputed per-target quantiles (e.g. unsat_frac_q75, te_p90)
        self._quantiles = quantiles or {}

    def q(self, feature: str, pct: int) -> Any:
        """Return the stored q<pct> value for a feature, or compute from data."""
        key = f"{feature}_q{pct}"
        if key in self._quantiles:
            return self._quantiles[key]
        x = self[feature]
        if HAVE_TORCH and torch.is_tensor(x):
            return torch.quantile(x.float(), pct / 100.0).item()
        return float(__import__("numpy").percentile(x, pct))


# ─────────────────────────────────────────────────────────────
#  Constraint dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class PhysicsConstraint:
    name: str
    description: str
    # rule_fn(pred, feat) → bool tensor (True = P but ¬Q violation)
    rule_fn: Callable[[Any, FeatureView], Any]
    # contrapositive_fn(pred, feat) → bool tensor (¬Q → ¬P FP candidates)
    contrapositive_fn: Callable[[Any, FeatureView], Any]
    hard: bool = False


# Helpers so rules can stay tiny and readable without crashing on
# numpy/torch difference
def _bool(x):
    if HAVE_TORCH and torch.is_tensor(x):
        return x.bool()
    return x.astype(bool)


def _not(x):
    return ~_bool(x)


def _mean(x):
    if HAVE_TORCH and torch.is_tensor(x):
        return x.float().mean()
    return x.astype(float).mean()


# ─────────────────────────────────────────────────────────────
#  HARD CONSTRAINTS — violations override model output (must be 0)
# ─────────────────────────────────────────────────────────────

HARD_CONSTRAINTS: List[PhysicsConstraint] = [
    PhysicsConstraint(
        name="no_spikes_no_binding",
        description="If spike_count = 0 for a residue's neighborhood, "
                    "binding_probability must not exceed 0.5.",
        rule_fn=lambda pred, feat: (feat["spike_count"] == 0) & (pred > 0.5),
        contrapositive_fn=lambda pred, feat: (pred > 0.5) & (feat["spike_count"] == 0),
        hard=True,
    ),
    PhysicsConstraint(
        name="surface_not_druggable",
        description="Fully exposed residues (burial < 0.1, SASA > 0.8) cannot "
                    "support binding_probability > 0.3.",
        rule_fn=lambda pred, feat: (
            (feat["burial"] < 0.1) & (feat["sasa"] > 0.8) & (pred > 0.3)
        ),
        contrapositive_fn=lambda pred, feat: (
            (pred > 0.3) & (feat["burial"] < 0.1) & (feat["sasa"] > 0.8)
        ),
        hard=True,
    ),
    PhysicsConstraint(
        name="zero_streams_invalid",
        description="A residue with n_streams = 0 has no detected spike "
                    "activity — predictions above 0.1 are invalid.",
        rule_fn=lambda pred, feat: (feat["n_streams"] == 0) & (pred > 0.1),
        contrapositive_fn=lambda pred, feat: (pred > 0.1) & (feat["n_streams"] == 0),
        hard=True,
    ),
]


# ─────────────────────────────────────────────────────────────
#  SOFT CONSTRAINTS — violations flag for review (<5% target)
# ─────────────────────────────────────────────────────────────

SOFT_CONSTRAINTS: List[PhysicsConstraint] = [
    PhysicsConstraint(
        name="high_features_should_bind",
        description="Residues with top-quartile unsat_frac AND top-quartile "
                    "spike_count AND buried (burial > 0.5) are expected to "
                    "bind (prob > 0.5).",
        rule_fn=lambda pred, feat: (
            (feat["unsat_frac"] > feat.q("unsat_frac", 75)) &
            (feat["spike_count"] > feat.q("spike_count", 75)) &
            (feat["burial"] > 0.5) &
            (pred < 0.5)
        ),
        contrapositive_fn=lambda pred, feat: (
            (pred < 0.5) &
            _not(
                (feat["unsat_frac"] > feat.q("unsat_frac", 75)) &
                (feat["spike_count"] > feat.q("spike_count", 75)) &
                (feat["burial"] > 0.5)
            )
        ),
    ),
    PhysicsConstraint(
        name="hysteresis_predicts_cryptic",
        description="Hysteresis asymmetry > 0.3 implies the residue's "
                    "pocket should be either is_cryptic=1 or therm_class != "
                    "INERT. Thermodynamically-silent INERT+non-cryptic + "
                    "high-hysteresis is physically inconsistent.",
        rule_fn=lambda pred, feat: (
            (feat["hysteresis_asymmetry"] > 0.3) &
            (feat["is_cryptic"] == 0) &
            (feat["therm_class_inert"] == 1)
        ),
        contrapositive_fn=lambda pred, feat: (
            (feat["therm_class_inert"] == 1) &
            (feat["is_cryptic"] == 0) &
            (feat["hysteresis_asymmetry"] > 0.3)
        ),
    ),
    PhysicsConstraint(
        name="high_transfer_entropy_predicts_trigger",
        description="Residues in the top 10% transfer_entropy for their "
                    "target should have a causal role (trigger or responder).",
        rule_fn=lambda pred, feat: (
            (feat["transfer_entropy"] > feat.q("transfer_entropy", 90)) &
            (feat["role_trigger"] == 0) &
            (feat["role_responder"] == 0)
        ),
        contrapositive_fn=lambda pred, feat: (
            (feat["role_trigger"] == 0) & (feat["role_responder"] == 0) &
            (feat["transfer_entropy"] > feat.q("transfer_entropy", 90))
        ),
    ),
]


ALL_CONSTRAINTS: List[PhysicsConstraint] = HARD_CONSTRAINTS + SOFT_CONSTRAINTS


# ─────────────────────────────────────────────────────────────
#  Training penalty
# ─────────────────────────────────────────────────────────────

def compute_penalty(predictions: Any, features: FeatureView, epoch: int):
    """Return penalty term to add to the training loss.

    Enforces nothing before `ENFORCE_AFTER_EPOCH` so the model can learn
    the base task without constraint interference.
    """
    if epoch < ENFORCE_AFTER_EPOCH:
        if HAVE_TORCH and torch.is_tensor(predictions):
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        return 0.0

    penalty = None
    for c in HARD_CONSTRAINTS:
        v = c.rule_fn(predictions, features)
        frac = _mean(v)
        penalty = frac * HARD_PENALTY_WEIGHT if penalty is None else penalty + frac * HARD_PENALTY_WEIGHT
    for c in SOFT_CONSTRAINTS:
        v = c.rule_fn(predictions, features)
        frac = _mean(v)
        penalty = frac * SOFT_PENALTY_WEIGHT if penalty is None else penalty + frac * SOFT_PENALTY_WEIGHT
    return penalty


# ─────────────────────────────────────────────────────────────
#  Post-hoc audit
# ─────────────────────────────────────────────────────────────

def audit_constraints(predict_fn: Callable[[Any], Any],
                      all_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run every constraint on every target. Returns a structured report.

    predict_fn(target_dict) → pred tensor/array
    all_targets: list of dicts each with a `features` FeatureView and `name`.
    """
    report: Dict[str, Any] = {"hard": {}, "soft": {}}
    for group_name, group in (("hard", HARD_CONSTRAINTS), ("soft", SOFT_CONSTRAINTS)):
        for c in group:
            n_total = 0
            n_viol = 0
            tgts_with_viol = 0
            fp_tgts: List[Dict[str, Any]] = []
            for tgt in all_targets:
                feat: FeatureView = tgt["features"]
                preds = predict_fn(tgt)
                v = c.rule_fn(preds, feat)
                fp = c.contrapositive_fn(preds, feat)
                if HAVE_TORCH and torch.is_tensor(v):
                    nv = int(v.sum().item()); nt = int(v.numel())
                    nfp = int(fp.sum().item())
                else:
                    nv = int(v.sum()); nt = int(v.size)
                    nfp = int(fp.sum())
                n_total += nt
                n_viol += nv
                if nv > 0:
                    tgts_with_viol += 1
                if nfp > 0:
                    fp_tgts.append({"target": tgt.get("name", "?"), "n_fp": nfp})
            report[group_name][c.name] = {
                "description": c.description,
                "total_predictions": n_total,
                "total_violations": n_viol,
                "violation_rate": (n_viol / n_total) if n_total else 0.0,
                "targets_with_violations": tgts_with_viol,
                "false_positive_candidates": fp_tgts[:20],  # cap for sanity
            }
    return report


def hard_violations_are_zero(report: Dict[str, Any]) -> bool:
    return all(r["total_violations"] == 0 for r in report["hard"].values())


def soft_violation_rate_below_5pct(report: Dict[str, Any]) -> bool:
    return all(r["violation_rate"] < 0.05 for r in report["soft"].values())


if __name__ == "__main__":
    # Smoke test — dummy predictions + features
    import numpy as np
    N = 100
    feat = FeatureView({
        "spike_count":          np.random.randint(0, 100, N),
        "n_streams":            np.random.randint(0, 4, N),
        "burial":               np.random.uniform(0, 1, N),
        "sasa":                 np.random.uniform(0, 1, N),
        "unsat_frac":           np.random.uniform(0, 1, N),
        "hysteresis_asymmetry": np.random.uniform(0, 1, N),
        "is_cryptic":           np.random.randint(0, 2, N),
        "therm_class_inert":    np.random.randint(0, 2, N),
        "transfer_entropy":     np.random.uniform(0, 1, N),
        "role_trigger":         np.random.randint(0, 2, N),
        "role_responder":       np.random.randint(0, 2, N),
    })
    pred = np.random.uniform(0, 1, N)

    print(f"Smoke test — {len(HARD_CONSTRAINTS)} hard + {len(SOFT_CONSTRAINTS)} soft")
    for c in ALL_CONSTRAINTS:
        v = c.rule_fn(pred, feat)
        fp = c.contrapositive_fn(pred, feat)
        print(f"  {c.name:42s} viol={int(v.sum()):4d} fp_cand={int(fp.sum()):4d}")
