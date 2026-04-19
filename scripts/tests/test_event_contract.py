"""CI test: event_schema_v1 contract is consistent with Worker W3b + D1 schema.

Pinned to v4 feature-service hardening contract, §4 enum forward-compat.
"""
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_event_schema_loadable():
    y = yaml.safe_load((REPO / "docs/contracts/event_schema_v1.yaml").read_text())
    assert y["schema_name"] == "event_schema_v1"
    assert y["schema_version"] == "1.0.0"


def test_ccns_phase_is_closed_enum():
    y = yaml.safe_load((REPO / "docs/contracts/event_schema_v1.yaml").read_text())
    phase = y["event_fields"]["ccns_phase"]
    assert phase["enum_open"] is False
    assert set(phase["enum"]) == {"cold_hold", "heating", "warm_hold", "cooling", "cold_return"}


def test_spike_source_is_closed_enum():
    y = yaml.safe_load((REPO / "docs/contracts/event_schema_v1.yaml").read_text())
    src = y["event_fields"]["spike_source"]
    assert src["enum_open"] is False
    assert set(src["enum"]) == {"UV", "LIF", "EFP"}


def test_type_is_open_enum():
    y = yaml.safe_load((REPO / "docs/contracts/event_schema_v1.yaml").read_text())
    t = y["event_fields"]["type"]
    assert t["enum_open"] is True
    assert "OTHER" not in t["enum"]   # OTHER is the bucket, not a declared enum value


def test_d1_schema_has_matching_count_columns():
    sql = (REPO / "cloudflare/d1/schema_phase4_site_tags.sql").read_text()
    # One column per enum value + one unknown/other bucket per enum.
    for col in ["count_phase_cold_hold", "count_phase_warm_hold", "count_phase_heating",
                "count_phase_cooling", "count_phase_cold_return", "count_phase_unknown",
                "count_source_uv", "count_source_lif", "count_source_efp", "count_source_other",
                "count_type_bnz", "count_type_unk", "count_type_anion", "count_type_cation",
                "count_type_other"]:
        assert col in sql, f"missing event-enum count column: {col}"


def test_worker_applies_quarantine_thresholds():
    text = (REPO / "cloudflare/workers/feature-pipeline/src/index.js").read_text()
    assert "THRESH_PHASE" in text
    assert "THRESH_SOURCE" in text
    assert "THRESH_TYPE" in text
    assert "quarantined_event_aggregates" in text
