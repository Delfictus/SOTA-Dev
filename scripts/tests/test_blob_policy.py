"""CI test: site_tags_json blob only holds decorative (non-R/F/M/T) keys.

Pinned to v4 feature-service hardening contract, §2 blob-only justification.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

BLOB_SAFE_KEYS = {
    "mean_burial", "asymmetry_offset",
    "sti_n_spikes", "sti_n_voxels",
    "composite_v3_score", "composite_audit_score",
    "composite_v3_rank", "composite_audit_rank", "cryptic_rank",
    "ranker_version", "tokenized_token",
    "tide_trigger_residues",
}


def test_blob_schema_membership_matches_contract():
    schema = json.loads((REPO / "docs/contracts/site_tags_json_v1.schema.json").read_text())
    allowed = set(schema["properties"].keys())
    extras = allowed - BLOB_SAFE_KEYS
    missing = BLOB_SAFE_KEYS - allowed
    assert not extras, f"blob schema has non-blob-safe keys: {sorted(extras)}"
    assert not missing, f"blob schema missing expected keys: {sorted(missing)}"


def test_blob_max_properties_bounded():
    schema = json.loads((REPO / "docs/contracts/site_tags_json_v1.schema.json").read_text())
    assert schema["maxProperties"] <= 13


def test_tokenized_token_dtype_is_integer_in_blob():
    schema = json.loads((REPO / "docs/contracts/site_tags_json_v1.schema.json").read_text())
    tt = schema["properties"]["tokenized_token"]
    # §3 dtype correction: tokenized_token is int, not string
    assert "integer" in tt["type"]
    assert "string" not in tt["type"]


def test_worker_enforces_blob_size_cap():
    text = (REPO / "cloudflare/workers/feature-pipeline/src/index.js").read_text()
    assert "SITE_TAGS_JSON_MAX_BYTES" in text
    assert "2048" in text   # 2 KB cap
