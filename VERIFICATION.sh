#!/usr/bin/env bash
set -euo pipefail

echo "=== PRISM-4D Hardened Release Verification ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

VERIFY_TMP="$(mktemp -d /tmp/prism-v025-verify.XXXXXX)"
trap 'rm -rf "$VERIFY_TMP"' EXIT
RELEASE_DIR="release_artifacts/v0.25.0"

echo "[1/7] Rust compilation and tests..."
cargo clippy -p prism-forge -- -D warnings
cargo test -p prism-forge --release
cargo check -p prism-nhs --bin warp_jacobian

echo "[2/7] Python type checking on hardened surfaces..."
PYTHONPATH=src:scripts python3 -m mypy --strict \
  src/prism_dstw/adapters/materials \
  src/prism_dstw/motif \
  scripts/audit_import_resolution.py \
  scripts/audit_schema_compatibility.py \
  scripts/audit_dependency_pinning.py \
  scripts/build_hardened_cbom.py \
  scripts/build_thermodynamic_motif_registry.py

echo "[3/7] Full test suite..."
PYTHONPATH=src:scripts python3 -m pytest tests/ -q

echo "[4/7] Import resolution audit..."
PYTHONPATH=src:scripts python3 scripts/audit_import_resolution.py \
  --output "$VERIFY_TMP/import_audit_report.json"
python3 - "$VERIFY_TMP/import_audit_report.json" "$RELEASE_DIR/import_audit_report.json" <<'PY'
import json
import sys
fresh = json.load(open(sys.argv[1], encoding="utf-8"))
sealed = json.load(open(sys.argv[2], encoding="utf-8"))
for key in ("resolved_count", "unresolved_count", "optional_unavailable_count", "parse_error_count", "circular_count"):
    assert fresh["summary"][key] == sealed["summary"][key], key
assert fresh["summary"]["unresolved_count"] == 0
assert fresh["summary"]["parse_error_count"] == 0
print("PASS: import audit matches sealed report")
PY

echo "[5/7] Schema compatibility audit..."
PYTHONPATH=src:scripts python3 scripts/audit_schema_compatibility.py \
  --output "$VERIFY_TMP/schema_compatibility_report.json"
python3 - "$VERIFY_TMP/schema_compatibility_report.json" "$RELEASE_DIR/schema_compatibility_report.json" <<'PY'
import json
import sys
fresh = json.load(open(sys.argv[1], encoding="utf-8"))
sealed = json.load(open(sys.argv[2], encoding="utf-8"))
assert fresh["summary"] == sealed["summary"]
assert fresh["summary"]["failure_count"] == 0
fresh_status = {row["name"]: row["status"] for row in fresh["contracts"]}
sealed_status = {row["name"]: row["status"] for row in sealed["contracts"]}
assert fresh_status == sealed_status
print("PASS: schema audit matches sealed report")
PY

echo "[6/7] Dependency pinning and CBOM..."
PYTHONPATH=src:scripts python3 scripts/audit_dependency_pinning.py \
  --output "$VERIFY_TMP/dependency_pinning_report.json"
python3 - "$VERIFY_TMP/dependency_pinning_report.json" "$RELEASE_DIR/dependency_pinning_report.json" <<'PY'
import json
import sys
fresh = json.load(open(sys.argv[1], encoding="utf-8"))
sealed = json.load(open(sys.argv[2], encoding="utf-8"))
assert fresh["summary"] == sealed["summary"]
assert fresh["summary"]["floating_python_count"] == 0
assert fresh["summary"]["cargo_lock_present"] is True
print("PASS: dependency audit matches sealed report")
PY
PYTHONPATH=src:scripts python3 scripts/build_hardened_cbom.py --output-dir "$VERIFY_TMP/cbom"
python3 - "$VERIFY_TMP/cbom/CBOM_v2.0.json" "$RELEASE_DIR/CBOM_v2.0.json" "$RELEASE_DIR/hardened_release_manifest.json" <<'PY'
import json
import sys
fresh = json.load(open(sys.argv[1], encoding="utf-8"))
sealed = json.load(open(sys.argv[2], encoding="utf-8"))
manifest = json.load(open(sys.argv[3], encoding="utf-8"))
assert fresh["root_hash"] == sealed["root_hash"]
assert set(fresh["subsystems"]) == set(sealed["subsystems"])
assert len(sealed["subsystems"]) == manifest["subsystem_count"]
assert len(sealed["subsystems"]) == 14
for name in sorted(sealed["subsystems"]):
    fresh_tree = fresh["subsystems"][name]
    sealed_tree = sealed["subsystems"][name]
    for key in ("root", "n_files", "total_size_bytes"):
        assert fresh_tree[key] == sealed_tree[key], (name, key)
print(f"PASS: CBOM root hash {sealed['root_hash']}")
PY

echo "[7/7] SHA256 checksum verification..."
sha256sum -c --quiet "$RELEASE_DIR/SHA256SUMS.txt"
echo "PASS: all release checksums match"

echo "=== VERIFICATION COMPLETE ==="
