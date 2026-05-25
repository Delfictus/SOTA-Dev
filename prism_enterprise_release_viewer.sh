#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/diddy/Desktop/Prism4D-bio"
CAMPAIGN="$ROOT/campaigns/glp1r_aleniglipron"
ARCHIVE="$ROOT/PRISM_GLP1R_M2_Release_v1.0.tar.gz"
SIG="$ROOT/PRISM_GLP1R_M2_Release_v1.0.tar.gz.sha256"
EXPECTED_SHA="59aa40b8d19e059cfdf9a928cda058044b19451a38ec44d39ad9e04b0f1f576e"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REVIEW_DIR="$ROOT/enterprise_release_review_$STAMP"
REPORT_MD="$REVIEW_DIR/ENTERPRISE_RELEASE_OBSERVABLE_REVIEW.md"
REPORT_HTML="$REVIEW_DIR/ENTERPRISE_RELEASE_VIEWER.html"
LOG="$REVIEW_DIR/review.log"

mkdir -p "$REVIEW_DIR"
cd "$ROOT"

exec > >(tee -a "$LOG") 2>&1

echo "======================================================================"
echo " PRISM-4D ENTERPRISE RELEASE VIEWER + AUDIT PORTAL"
echo "======================================================================"
echo "ROOT:      $ROOT"
echo "CAMPAIGN:  $CAMPAIGN"
echo "REVIEW:    $REVIEW_DIR"
echo

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[FATAL] Missing required command: $1"
    exit 1
  }
}

require python3
require jq
require sha256sum
require tar
require find
require awk
require sed
require grep

cat > "$REPORT_MD" <<EOF
# PRISM-4D GLP-1R M2 Enterprise Observable Release Review

Generated UTC: \`$(date -u --iso-8601=seconds)\`

Root: \`$ROOT\`  
Campaign: \`$CAMPAIGN\`  
Review directory: \`$REVIEW_DIR\`

EOF

section() {
  echo
  echo "======================================================================"
  echo " $1"
  echo "======================================================================"
  echo
  echo "" >> "$REPORT_MD"
  echo "## $1" >> "$REPORT_MD"
  echo "" >> "$REPORT_MD"
}

ok_md() {
  echo "- [OK] $1" >> "$REPORT_MD"
  echo "[OK] $1"
}

warn_md() {
  echo "- [WARN] $1" >> "$REPORT_MD"
  echo "[WARN] $1"
}

fail_md() {
  echo "- [FAIL] $1" >> "$REPORT_MD"
  echo "[FAIL] $1"
}

# ---------------------------------------------------------------------
section "1. Required Release Artifact Presence"

declare -A REQUIRED_FILES=(
  ["Release archive"]="$ARCHIVE"
  ["Release SHA256 signature"]="$SIG"
  ["Master Data Room Index"]="$CAMPAIGN/MASTER_DATA_ROOM_INDEX.md"
  ["Pharmacological Dynamics Intelligence Report"]="$CAMPAIGN/M2_Pharmacological_Dynamics_Intelligence_Report.md"
  ["Triangulation Dossier"]="$CAMPAIGN/M2_Triangulation_Dossier_Final.md"
  ["Executive Readout Final"]="$CAMPAIGN/M2_Executive_Readout_Final.md"
  ["Enterprise Positioning Summary"]="$CAMPAIGN/ENTERPRISE_POSITIONING_SUMMARY.md"
  ["Claim Falsification Graph"]="$CAMPAIGN/claim_falsification_graph.json"
  ["CBOM"]="$CAMPAIGN/PRISM_CBOM_v1.0.json"
  ["Replayability Manifest"]="$CAMPAIGN/M2_Replayability_Manifest.json"
  ["Metastable Atlas Triggers"]="$CAMPAIGN/phase_2c_metastable_atlas_triggers.json"
  ["Snapshot Triggers"]="$CAMPAIGN/phase_2c_snapshot_triggers.json"
  ["CRO WetLab Action Plan"]="$CAMPAIGN/CRO_WetLab_Action_Plan.parquet"
  ["Zero-shot Teaser Solutions"]="$CAMPAIGN/track_0_manual_emulation/teaser_solutions.parquet"
)

for label in "${!REQUIRED_FILES[@]}"; do
  path="${REQUIRED_FILES[$label]}"
  if [[ -f "$path" ]]; then
    size="$(du -h "$path" | awk '{print $1}')"
    ok_md "$label exists: \`$path\` size=\`$size\`"
  else
    fail_md "$label missing: \`$path\`"
  fi
done

if [[ -d "$CAMPAIGN/visualizer_app" && -f "$CAMPAIGN/visualizer_app/index.html" ]]; then
  ok_md "Visualizer static build exists: \`$CAMPAIGN/visualizer_app/index.html\`"
else
  fail_md "Visualizer static build missing or incomplete."
fi

# ---------------------------------------------------------------------
section "2. Immutable Archive Verification"

if [[ -f "$ARCHIVE" ]]; then
  ACTUAL_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  ARCHIVE_SIZE="$(du -h "$ARCHIVE" | awk '{print $1}')"

  echo "- Archive: \`$ARCHIVE\`" >> "$REPORT_MD"
  echo "- Archive size: \`$ARCHIVE_SIZE\`" >> "$REPORT_MD"
  echo "- Actual SHA-256: \`$ACTUAL_SHA\`" >> "$REPORT_MD"
  echo "- Expected SHA-256: \`$EXPECTED_SHA\`" >> "$REPORT_MD"

  if [[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]]; then
    ok_md "Archive hash matches expected release fingerprint."
  else
    warn_md "Archive hash does not match hardcoded expected fingerprint. Check whether archive was rebuilt."
  fi

  if [[ -f "$SIG" ]]; then
    if sha256sum -c "$SIG"; then
      ok_md "Detached SHA256 signature verifies successfully."
    else
      fail_md "Detached SHA256 signature verification failed."
    fi
  fi
fi

# ---------------------------------------------------------------------
section "3. CBOM / Merkle / Environment Fingerprint"

python3 - <<PY
import json
from pathlib import Path

cbom_path = Path("$CAMPAIGN/PRISM_CBOM_v1.0.json")
report = Path("$REPORT_MD")

if not cbom_path.exists():
    raise SystemExit

cbom = json.loads(cbom_path.read_text())

root = cbom.get("campaign_merkle_root") or cbom.get("merkle_root")
file_count = cbom.get("file_count")
directory_count = cbom.get("directory_count")
schema = cbom.get("schema") or cbom.get("cbom_schema") or cbom.get("CBOM schema")
env = cbom.get("environment_fingerprint") or {}

print("CBOM root:", root)
print("file_count:", file_count)
print("directory_count:", directory_count)
print("environment:", env)

with report.open("a") as f:
    f.write(f"- CBOM Merkle root: `{root}`\\n")
    f.write(f"- File count: `{file_count}`\\n")
    f.write(f"- Directory count: `{directory_count}`\\n")
    f.write(f"- Schema: `{schema}`\\n")
    f.write("- Environment fingerprint:\\n")
    if isinstance(env, dict) and env:
        for k, v in env.items():
            f.write(f"  - `{k}`: `{v}`\\n")
    else:
        f.write("  - `[WARN] environment_fingerprint missing or null`\\n")

required = ["python", "polars", "platform", "os_kernel"]
missing = [k for k in required if not isinstance(env, dict) or not env.get(k)]
if missing:
    print("[WARN] Missing env fields:", missing)
    with report.open("a") as f:
        f.write(f"- [WARN] Missing core environment fields: `{missing}`\\n")
else:
    print("[OK] Core environment fields present")
    with report.open("a") as f:
        f.write("- [OK] Core environment fields present.\\n")
PY

# ---------------------------------------------------------------------
section "4. JSON Structural Validation And Summaries"

JSONS=(
  "$CAMPAIGN/claim_falsification_graph.json"
  "$CAMPAIGN/PRISM_CBOM_v1.0.json"
  "$CAMPAIGN/M2_Replayability_Manifest.json"
  "$CAMPAIGN/phase_2c_metastable_atlas_triggers.json"
  "$CAMPAIGN/phase_2c_snapshot_triggers.json"
  "$CAMPAIGN/phase_2c_reintegration_parity.json"
)

for f in "${JSONS[@]}"; do
  [[ -f "$f" ]] || continue
  if jq empty "$f"; then
    ok_md "JSON valid: \`$f\`"
    echo "" >> "$REPORT_MD"
    echo "Top-level keys for \`$f\`:" >> "$REPORT_MD"
    echo '```json' >> "$REPORT_MD"
    jq 'keys' "$f" >> "$REPORT_MD"
    echo '```' >> "$REPORT_MD"
  else
    fail_md "JSON invalid: \`$f\`"
  fi
done

# ---------------------------------------------------------------------
section "5. Parquet Observable Review"

python3 - <<PY
from pathlib import Path
import json
import polars as pl

campaign = Path("$CAMPAIGN")
report = Path("$REPORT_MD")
out_dir = Path("$REVIEW_DIR/parquet_samples")
out_dir.mkdir(parents=True, exist_ok=True)

important = [
    campaign / "CRO_WetLab_Action_Plan.parquet",
    campaign / "track_0_manual_emulation/teaser_solutions.parquet",
    campaign / "track_0_manual_emulation/fragment_interference_attribution.parquet",
    campaign / "integrated_spike_events/n80_full_scale/phase_manifold_coherence.parquet",
    campaign / "integrated_spike_events/n80_full_scale/phase_manifold_edge_validation.parquet",
    campaign / "integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet",
    campaign / "integrated_spike_events/n80_full_scale/hysteresis_tensor.parquet",
    campaign / "integrated_spike_events/n80_full_scale/temporal_cascade.parquet",
    campaign / "integrated_spike_events/n80_full_scale/assay_routing_recommendations.parquet",
    campaign / "track_a_generative/calibration_anchors_3d.parquet",
    campaign / "track_a_generative/gflownet_tso_bridge_boundaries.parquet",
]

with report.open("a") as f:
    f.write("| Artifact | Rows/sample | Columns | Status |\\n")
    f.write("|---|---:|---:|---|\\n")

for p in important:
    if not p.exists():
        with report.open("a") as f:
            f.write(f"| `{p}` | 0 | 0 | MISSING |\\n")
        continue
    try:
        df = pl.read_parquet(p, n_rows=20)
        schema = {k: str(v) for k, v in df.schema.items()}
        sample_name = p.name.replace(".parquet", ".sample.tsv")
        sample_path = out_dir / sample_name
        df.write_csv(sample_path, separator="\\t")

        with report.open("a") as f:
            f.write(f"| `{p.relative_to(campaign)}` | {df.height} sample | {df.width} | OK; sample `{sample_path}` |\\n")
            f.write("\\n")
            f.write(f"Schema for `{p.relative_to(campaign)}`:\\n")
            f.write("```json\\n")
            f.write(json.dumps(schema, indent=2))
            f.write("\\n```\\n\\n")

        print(f"[OK] {p}: sample_rows={df.height}, cols={df.width}")
    except Exception as e:
        with report.open("a") as f:
            f.write(f"| `{p}` | 0 | 0 | FAIL: {e} |\\n")
        print(f"[FAIL] {p}: {e}")
PY

# ---------------------------------------------------------------------
section "6. Master Data Room Deep Content Scan"

MASTER="$CAMPAIGN/MASTER_DATA_ROOM_INDEX.md"

if [[ -f "$MASTER" ]]; then
  echo "Master Index headings:" >> "$REPORT_MD"
  echo '```text' >> "$REPORT_MD"
  grep -n '^## ' "$MASTER" >> "$REPORT_MD" || true
  echo '```' >> "$REPORT_MD"

  for term in \
    "Executive Dossier" \
    "Interactive 3D Visualizer" \
    "Medicinal Chemistry Action Center" \
    "Wet-Lab CRO Handoff" \
    "Track A Cloud AI Readiness" \
    "Pending GPU Campaigns" \
    "Cryptographic Bill of Materials"
  do
    if grep -q "$term" "$MASTER"; then
      ok_md "Master Data Room contains section: $term"
    else
      warn_md "Master Data Room missing expected section: $term"
    fi
  done

  if grep -n "|  |  |" "$MASTER" > "$REVIEW_DIR/blank_table_hits.txt"; then
    warn_md "Blank table cells detected in Master Data Room. See \`$REVIEW_DIR/blank_table_hits.txt\`"
  else
    ok_md "No obvious blank markdown table rows detected."
  fi
fi

# ---------------------------------------------------------------------
section "7. Overclaim / False Precision / Legal-Risk Language Scan"

PATTERNS=(
  "32 femtosecond"
  "irreversible desensitization"
  "confirmed biological efficacy"
  "proves clinical"
  "guarantees"
  "guaranteed"
  "expected Δ uptake > 15"
  "Expected Δ uptake > 15"
  "expected fixed HDX"
  "therapeutic efficacy"
  "patient response"
  "clinical outcome"
  "mechanistic proof"
)

: > "$REVIEW_DIR/overclaim_hits.txt"

for pat in "${PATTERNS[@]}"; do
  if grep -RIn --exclude="*.tar.gz" --exclude-dir="visualizer_app/assets" --exclude-dir="node_modules" "$pat" "$CAMPAIGN" >> "$REVIEW_DIR/overclaim_hits.txt" 2>/dev/null; then
    warn_md "Possible overclaim pattern found: \`$pat\`"
  else
    ok_md "No overclaim pattern found: \`$pat\`"
  fi
done

if [[ -s "$REVIEW_DIR/overclaim_hits.txt" ]]; then
  echo "" >> "$REPORT_MD"
  echo "Overclaim hits:" >> "$REPORT_MD"
  echo '```text' >> "$REPORT_MD"
  head -80 "$REVIEW_DIR/overclaim_hits.txt" >> "$REPORT_MD"
  echo '```' >> "$REPORT_MD"
fi

# ---------------------------------------------------------------------
section "8. Claim Falsification Graph Review"

if [[ -f "$CAMPAIGN/claim_falsification_graph.json" ]]; then
  echo "First claims:" >> "$REPORT_MD"
  echo '```json' >> "$REPORT_MD"
  jq '.claims[0:10]' "$CAMPAIGN/claim_falsification_graph.json" >> "$REPORT_MD"
  echo '```' >> "$REPORT_MD"
fi

# ---------------------------------------------------------------------
section "9. CRO Handoff Full Observable Table"

python3 - <<PY
from pathlib import Path
import polars as pl

p = Path("$CAMPAIGN/CRO_WetLab_Action_Plan.parquet")
report = Path("$REPORT_MD")

if p.exists():
    df = pl.read_parquet(p)
    with report.open("a") as f:
        f.write(f"- Rows: `{df.height}`\\n")
        f.write(f"- Columns: `{df.columns}`\\n\\n")
        f.write("```text\\n")
        f.write(str(df))
        f.write("\\n```\\n")
PY

# ---------------------------------------------------------------------
section "10. Medicinal Chemistry / Zero-Shot Replacement Full Review"

python3 - <<PY
from pathlib import Path
import polars as pl

p = Path("$CAMPAIGN/track_0_manual_emulation/teaser_solutions.parquet")
report = Path("$REPORT_MD")

if p.exists():
    df = pl.read_parquet(p)
    cols = [
        "solution_rank",
        "anchor_id",
        "canonical_smiles",
        "sa_score",
        "pi_complement",
        "pi_clash",
        "projected_durability_improvement",
        "liability_edge_label",
        "solution_epistemic_class",
    ]
    cols = [c for c in cols if c in df.columns]
    view = df.select(cols)

    with report.open("a") as f:
        f.write(f"- Rows: `{df.height}`\\n")
        f.write(f"- Columns: `{df.columns}`\\n\\n")
        f.write("```text\\n")
        f.write(str(view))
        f.write("\\n```\\n")
PY

# ---------------------------------------------------------------------
section "11. Metastable / Chronology Infrastructure Review"

if [[ -f "$CAMPAIGN/phase_2c_metastable_atlas_triggers.json" ]]; then
  echo "Metastable trigger count:" >> "$REPORT_MD"
  echo '```json' >> "$REPORT_MD"
  jq '{trigger_count, capture_mode, first_triggers: .triggers[0:5]}' "$CAMPAIGN/phase_2c_metastable_atlas_triggers.json" >> "$REPORT_MD"
  echo '```' >> "$REPORT_MD"
fi

if [[ -f "$CAMPAIGN/phase_2c_snapshot_triggers.json" ]]; then
  echo "Snapshot trigger summary:" >> "$REPORT_MD"
  echo '```json' >> "$REPORT_MD"
  jq 'del(.triggers)' "$CAMPAIGN/phase_2c_snapshot_triggers.json" >> "$REPORT_MD" || true
  echo '```' >> "$REPORT_MD"
fi

# ---------------------------------------------------------------------
section "12. Holo Topology / Pending GPU Campaign Review"

HOLO="$ROOT/04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json"

if [[ -f "$HOLO" ]]; then
  ok_md "Holo topology exists: \`$HOLO\`"
  echo '```json' >> "$REPORT_MD"
  jq '{n_atoms, ligand_atoms, ligand_charge_method, min_heavy_distance_A, selected_source_condition}' "$HOLO" >> "$REPORT_MD" || true
  echo '```' >> "$REPORT_MD"
else
  warn_md "Holo topology not found at expected absolute path: \`$HOLO\`"
fi

if [[ -f "$ROOT/bin/launch-n80-holo-aleniglipron.sh" ]]; then
  ok_md "Holo launch script exists: \`$ROOT/bin/launch-n80-holo-aleniglipron.sh\`"
fi

# ---------------------------------------------------------------------
section "13. Track A / Cloud AI Readiness Review"

TRACK_A="$CAMPAIGN/track_a_generative"

if [[ -d "$TRACK_A" ]]; then
  echo "Track A artifacts:" >> "$REPORT_MD"
  echo '```text' >> "$REPORT_MD"
  find "$TRACK_A" -maxdepth 2 -type f -printf "%p %s bytes\n" | sort >> "$REPORT_MD"
  echo '```' >> "$REPORT_MD"
fi

CLOUDFLARE="$ROOT/00_registry/architecture/Cloudflare_Manifold_Architecture.md"
if [[ -f "$CLOUDFLARE" ]]; then
  ok_md "Cloudflare architecture exists: \`$CLOUDFLARE\`"
  echo "Cloudflare architecture headings:" >> "$REPORT_MD"
  echo '```text' >> "$REPORT_MD"
  grep -n '^#' "$CLOUDFLARE" >> "$REPORT_MD" || true
  echo '```' >> "$REPORT_MD"
fi

# ---------------------------------------------------------------------
section "14. Full Campaign Inventory"

find "$CAMPAIGN" -type f -printf "%p\t%s bytes\n" | sort > "$REVIEW_DIR/full_campaign_inventory.tsv"

echo "- Full live campaign inventory: \`$REVIEW_DIR/full_campaign_inventory.tsv\`" >> "$REPORT_MD"
echo "- File count in live campaign folder: \`$(wc -l < "$REVIEW_DIR/full_campaign_inventory.tsv")\`" >> "$REPORT_MD"

echo '```text' >> "$REPORT_MD"
head -200 "$REVIEW_DIR/full_campaign_inventory.tsv" >> "$REPORT_MD"
echo '```' >> "$REPORT_MD"

# ---------------------------------------------------------------------
section "15. Archive Inventory Observable Sample"

if [[ -f "$ARCHIVE" ]]; then
  echo "Reading archive inventory sample. This may take time on a 100G gzip..."
  tar -tzf "$ARCHIVE" | head -200 > "$REVIEW_DIR/archive_inventory_head.txt" || true

  echo "- Archive inventory sample: \`$REVIEW_DIR/archive_inventory_head.txt\`" >> "$REPORT_MD"
  echo '```text' >> "$REPORT_MD"
  cat "$REVIEW_DIR/archive_inventory_head.txt" >> "$REPORT_MD"
  echo '```' >> "$REPORT_MD"

  tar -tzf "$ARCHIVE" | grep -E '\.(dcd|xtc|trr|nc|raw|bin)$' | head -50 > "$REVIEW_DIR/possible_raw_bulk_files.txt" || true

  if [[ -s "$REVIEW_DIR/possible_raw_bulk_files.txt" ]]; then
    warn_md "Possible raw/bulk trajectory files found in release archive. See \`$REVIEW_DIR/possible_raw_bulk_files.txt\`"
  else
    ok_md "No obvious raw trajectory/bulk binary files found in archive scan."
  fi
fi

# ---------------------------------------------------------------------
section "16. Build HTML Viewer Portal"

python3 - <<PY
from pathlib import Path
import html

review = Path("$REVIEW_DIR")
md = Path("$REPORT_MD").read_text(errors="replace")
campaign = Path("$CAMPAIGN")
root = Path("$ROOT")

def block(title: str, body: str) -> str:
    return f"<section><h2>{html.escape(title)}</h2><pre>{html.escape(body)}</pre></section>"

links = [
    ("Master Data Room", campaign / "MASTER_DATA_ROOM_INDEX.md"),
    ("Pharmacological Dynamics Intelligence Report", campaign / "M2_Pharmacological_Dynamics_Intelligence_Report.md"),
    ("Triangulation Dossier", campaign / "M2_Triangulation_Dossier_Final.md"),
    ("Executive Readout", campaign / "M2_Executive_Readout_Final.md"),
    ("Enterprise Positioning Summary", campaign / "ENTERPRISE_POSITIONING_SUMMARY.md"),
    ("Claim Falsification Graph", campaign / "claim_falsification_graph.json"),
    ("CBOM", campaign / "PRISM_CBOM_v1.0.json"),
    ("Replayability Manifest", campaign / "M2_Replayability_Manifest.json"),
    ("Visualizer", campaign / "visualizer_app/index.html"),
]

link_html = "\\n".join(
    f'<li><a href="file://{p}">{html.escape(name)}</a></li>'
    for name, p in links
    if p.exists()
)

html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PRISM GLP-1R M2 Enterprise Release Viewer</title>
<style>
body {{
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 32px;
  background: #0f1117;
  color: #e7e7e7;
}}
a {{ color: #8ab4ff; }}
pre {{
  background: #171a22;
  border: 1px solid #333846;
  border-radius: 12px;
  padding: 16px;
  white-space: pre-wrap;
  overflow-x: auto;
}}
section {{
  margin-bottom: 32px;
}}
.badge {{
  display: inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  background: #25304a;
  color: #bcd2ff;
  font-size: 12px;
}}
</style>
</head>
<body>
<h1>PRISM GLP-1R M2 Enterprise Release Viewer</h1>
<p><span class="badge">Immutable Release</span> <span class="badge">Epistemic Hardening</span> <span class="badge">CRO Falsification Gates</span> <span class="badge">CBOM</span></p>

<h2>Direct Artifact Links</h2>
<ul>
{link_html}
</ul>

<h2>Visualizer</h2>
<p>Static visualizer path: <code>{campaign / "visualizer_app/index.html"}</code></p>
<p>HTTP viewer will be served at: <a href="http://127.0.0.1:8080">http://127.0.0.1:8080</a></p>

<h2>Review Report</h2>
<pre>{html.escape(md)}</pre>
</body>
</html>
"""

Path("$REPORT_HTML").write_text(html_doc)
PY

ok_md "HTML viewer generated: \`$REPORT_HTML\`"

# ---------------------------------------------------------------------
section "17. Launch Local Servers"

# Visualizer server on 8080
if [[ -f "$CAMPAIGN/visualizer_app/index.html" ]]; then
  if ! ss -ltn | grep -q ':8080 '; then
    (cd "$CAMPAIGN/visualizer_app" && python3 -m http.server 8080 >/tmp/prism_visualizer_8080.log 2>&1 &)
    ok_md "Visualizer server started at http://127.0.0.1:8080"
  else
    ok_md "Port 8080 already active; visualizer may already be running."
  fi
fi

# Review portal server on 8090
if ! ss -ltn | grep -q ':8090 '; then
  (cd "$REVIEW_DIR" && python3 -m http.server 8090 >/tmp/prism_review_8090.log 2>&1 &)
  ok_md "Enterprise review portal started at http://127.0.0.1:8090/ENTERPRISE_RELEASE_VIEWER.html"
else
  ok_md "Port 8090 already active; review portal may already be running."
fi

# ---------------------------------------------------------------------
section "18. Final Human Review Instructions"

cat >> "$REPORT_MD" <<EOF

Open these in browser:

- Enterprise release viewer: http://127.0.0.1:8090/ENTERPRISE_RELEASE_VIEWER.html
- Interactive visualizer: http://127.0.0.1:8080

Terminal review commands:

\`\`\`bash
less "$REPORT_MD"
less "$CAMPAIGN/MASTER_DATA_ROOM_INDEX.md"
jq . "$CAMPAIGN/PRISM_CBOM_v1.0.json" | less
jq . "$CAMPAIGN/claim_falsification_graph.json" | less
\`\`\`

Critical known review flag:

- If \`blank_table_hits.txt\` exists, fix or remove blank Dynamic Pharmacophore/SAR rows before external delivery.

EOF

echo
echo "======================================================================"
echo " ENTERPRISE RELEASE VIEWER COMPLETE"
echo "======================================================================"
echo
echo "Markdown report:"
echo "  $REPORT_MD"
echo
echo "HTML portal:"
echo "  $REPORT_HTML"
echo
echo "Browser URLs:"
echo "  http://127.0.0.1:8090/ENTERPRISE_RELEASE_VIEWER.html"
echo "  http://127.0.0.1:8080"
echo
echo "Open terminal report:"
echo "  less \"$REPORT_MD\""
echo

if [[ -f "$REVIEW_DIR/blank_table_hits.txt" ]]; then
  echo "[FINAL FLAG] Blank table cells detected. Fix before external release:"
  echo "  $REVIEW_DIR/blank_table_hits.txt"
else
  echo "[FINAL STATUS] No obvious blank tables detected."
fi
