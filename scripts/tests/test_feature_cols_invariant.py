"""CI test: v4 FEATURE_COLS is exactly 15 entries in the pinned order.

Pinned to v4 feature-service hardening contract, §4 FEATURE_COLS correction.
"""
import hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/training"))

EXPECTED = [
    "spike_count", "n_streams", "interaction", "unsat_frac", "persistence",
    "log_spike_count", "log_interaction", "spread", "burial_score", "spike_density",
    "druggability", "aromatic_score", "n_lining_residues",
    "phase_transition_ratio", "warm_hold_spike_fraction",
]


def _canonical_hash(cols):
    return hashlib.sha256(
        json.dumps(cols, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def test_feature_cols_length_is_15():
    import xgboost_ranker_v4 as v4
    assert len(v4.FEATURE_COLS) == 15, f"got {len(v4.FEATURE_COLS)}"


def test_feature_cols_order_matches():
    import xgboost_ranker_v4 as v4
    assert list(v4.FEATURE_COLS) == EXPECTED


def test_feature_cols_contract_yaml_matches():
    import yaml
    contract = yaml.safe_load((REPO / "scripts/training/v4_feature_contract.yaml").read_text())
    assert contract["feature_cols"]["count"] == 15
    assert contract["feature_cols"]["ordered_list"] == EXPECTED


def test_feature_cols_hash_is_deterministic():
    import xgboost_ranker_v4 as v4
    assert _canonical_hash(list(v4.FEATURE_COLS)) == _canonical_hash(EXPECTED)
