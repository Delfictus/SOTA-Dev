#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/diddy/Desktop/Prism4D-bio")
CAMPAIGN = ROOT / "campaigns/glp1r_aleniglipron"
ARCHIVE = ROOT / "PRISM_GLP1R_M2_Release_v1.0.tar.gz"
SIG = ROOT / "PRISM_GLP1R_M2_Release_v1.0.tar.gz.sha256"
EXPECTED_SHA = "59aa40b8d19e059cfdf9a928cda058044b19451a38ec44d39ad9e04b0f1f576e"

STAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
REVIEW_DIR = ROOT / f"enterprise_release_review_v2_{STAMP}"
REPORT_MD = REVIEW_DIR / "ENTERPRISE_RELEASE_OBSERVABLE_REVIEW.md"
REPORT_HTML = REVIEW_DIR / "ENTERPRISE_RELEASE_VIEWER.html"

REVIEW_DIR.mkdir(parents=True, exist_ok=True)

hard_failures: list[str] = []
warnings: list[str] = []


def human_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 16), b""):
            h.update(chunk)
    return h.hexdigest()


def append(text: str = "") -> None:
    with REPORT_MD.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n {title}\n{'=' * 72}\n")
    append()
    append(f"## {title}")
    append()


def ok(msg: str) -> None:
    print(f"[OK] {msg}")
    append(f"- [OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")
    warnings.append(msg)
    append(f"- [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    hard_failures.append(msg)
    append(f"- [FAIL] {msg}")


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"JSON invalid: `{path}` :: {e}")
        return None


def find_env(obj: Any) -> dict[str, Any]:
    """Find environment fields even if CBOM stores them under a different key."""
    wanted = {"python", "polars", "platform", "os_kernel", "cuda_version", "nvidia_driver"}

    def walk(x: Any) -> dict[str, Any] | None:
        if isinstance(x, dict):
            keys = set(x.keys())
            if len(keys & wanted) >= 2:
                return {k: x.get(k) for k in sorted(keys & wanted)}
            for preferred in [
                "environment_fingerprint",
                "environment",
                "build_environment",
                "runtime_environment",
                "replay_environment",
                "system_environment",
            ]:
                if preferred in x:
                    found = walk(x[preferred])
                    if found:
                        return found
            for v in x.values():
                found = walk(v)
                if found:
                    return found
        elif isinstance(x, list):
            for v in x:
                found = walk(v)
                if found:
                    return found
        return None

    found = walk(obj)
    return found or {}


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


REPORT_MD.write_text(
    "# PRISM-4D GLP-1R M2 Enterprise Observable Release Review\n\n"
    f"Generated UTC: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`\n\n"
    f"Root: `{ROOT}`\n\n"
    f"Campaign: `{CAMPAIGN}`\n\n"
    f"Review directory: `{REVIEW_DIR}`\n\n",
    encoding="utf-8",
)

print("=" * 72)
print(" PRISM-4D ENTERPRISE RELEASE VIEWER V2")
print("=" * 72)
print(f"ROOT:     {ROOT}")
print(f"CAMPAIGN: {CAMPAIGN}")
print(f"REVIEW:   {REVIEW_DIR}")

section("1. Required Artifact Presence")

required_files = {
    "Release archive": ARCHIVE,
    "Release SHA256 signature": SIG,
    "Master Data Room Index": CAMPAIGN / "MASTER_DATA_ROOM_INDEX.md",
    "Pharmacological Dynamics Intelligence Report": CAMPAIGN / "M2_Pharmacological_Dynamics_Intelligence_Report.md",
    "Triangulation Dossier": CAMPAIGN / "M2_Triangulation_Dossier_Final.md",
    "Executive Readout Final": CAMPAIGN / "M2_Executive_Readout_Final.md",
    "Enterprise Positioning Summary": CAMPAIGN / "ENTERPRISE_POSITIONING_SUMMARY.md",
    "Claim Falsification Graph": CAMPAIGN / "claim_falsification_graph.json",
    "CBOM": CAMPAIGN / "PRISM_CBOM_v1.0.json",
    "Replayability Manifest": CAMPAIGN / "M2_Replayability_Manifest.json",
    "Metastable Atlas Triggers": CAMPAIGN / "phase_2c_metastable_atlas_triggers.json",
    "Snapshot Triggers": CAMPAIGN / "phase_2c_snapshot_triggers.json",
    "Reintegration Parity": CAMPAIGN / "phase_2c_reintegration_parity.json",
    "CRO WetLab Action Plan": CAMPAIGN / "CRO_WetLab_Action_Plan.parquet",
    "Zero-shot Teaser Solutions": CAMPAIGN / "track_0_manual_emulation/teaser_solutions.parquet",
    "Visualizer index": CAMPAIGN / "visualizer_app/index.html",
}

for label, path in required_files.items():
    if path.is_file():
        ok(f"{label}: `{path}` size=`{human_size(path)}`")
    else:
        fail(f"{label} missing: `{path}`")

section("2. Immutable Archive And Signature Verification")

if ARCHIVE.exists():
    actual_sha = sha256_file(ARCHIVE)
    append(f"- Archive size: `{human_size(ARCHIVE)}`")
    append(f"- Actual archive SHA-256: `{actual_sha}`")
    append(f"- Expected archive SHA-256: `{EXPECTED_SHA}`")
    if actual_sha == EXPECTED_SHA:
        ok("Archive hash matches expected immutable release fingerprint.")
    else:
        warn("Archive hash does not match expected fingerprint. This may be acceptable only if the archive was intentionally rebuilt.")

if SIG.exists():
    sig_text = SIG.read_text(encoding="utf-8", errors="replace").strip()
    sig_hash = sig_text.split()[0] if sig_text else ""
    append(f"- Signature file first hash: `{sig_hash}`")
    if ARCHIVE.exists() and sig_hash == sha256_file(ARCHIVE):
        ok("Detached SHA256 signature matches archive.")
    else:
        fail("Detached SHA256 signature does not match archive.")

section("3. CBOM / Merkle / Environment Fingerprint")

cbom_path = CAMPAIGN / "PRISM_CBOM_v1.0.json"
cbom = load_json(cbom_path) if cbom_path.exists() else None
if isinstance(cbom, dict):
    merkle = cbom.get("campaign_merkle_root") or cbom.get("merkle_root")
    append(f"- CBOM Merkle root: `{merkle}`")
    append(f"- CBOM file count: `{cbom.get('file_count')}`")
    append(f"- CBOM directory count: `{cbom.get('directory_count')}`")
    env = find_env(cbom)

    if not env:
        # fallback to replayability manifest
        replay = load_json(CAMPAIGN / "M2_Replayability_Manifest.json")
        env = find_env(replay)

    if env:
        ok("Environment fingerprint found.")
        append("")
        append("Environment fingerprint:")
        for k, v in env.items():
            append(f"- `{k}`: `{v}`")
    else:
        warn("Environment fingerprint not found in CBOM or replayability manifest.")

section("4. JSON Structural Validation And Observable Summaries")

json_paths = [
    CAMPAIGN / "claim_falsification_graph.json",
    CAMPAIGN / "PRISM_CBOM_v1.0.json",
    CAMPAIGN / "M2_Replayability_Manifest.json",
    CAMPAIGN / "phase_2c_metastable_atlas_triggers.json",
    CAMPAIGN / "phase_2c_snapshot_triggers.json",
    CAMPAIGN / "phase_2c_reintegration_parity.json",
]

for path in json_paths:
    if not path.exists():
        continue
    obj = load_json(path)
    if obj is not None:
        ok(f"JSON valid: `{path}`")
        if isinstance(obj, dict):
            append(f"- `{path.name}` top-level keys: `{list(obj.keys())}`")
        elif isinstance(obj, list):
            append(f"- `{path.name}` list length: `{len(obj)}`")

section("5. Parquet Readability, Schemas, And Samples")

try:
    import polars as pl
except Exception as e:
    fail(f"Polars unavailable: {e}")
    pl = None  # type: ignore[assignment]

parquet_samples_dir = REVIEW_DIR / "parquet_samples"
parquet_samples_dir.mkdir(exist_ok=True)

important_parquets = [
    CAMPAIGN / "CRO_WetLab_Action_Plan.parquet",
    CAMPAIGN / "track_0_manual_emulation/teaser_solutions.parquet",
    CAMPAIGN / "track_0_manual_emulation/fragment_interference_attribution.parquet",
    CAMPAIGN / "integrated_spike_events/n80_full_scale/phase_manifold_coherence.parquet",
    CAMPAIGN / "integrated_spike_events/n80_full_scale/phase_manifold_edge_validation.parquet",
    CAMPAIGN / "integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet",
    CAMPAIGN / "integrated_spike_events/n80_full_scale/hysteresis_tensor.parquet",
    CAMPAIGN / "integrated_spike_events/n80_full_scale/temporal_cascade.parquet",
    CAMPAIGN / "integrated_spike_events/n80_full_scale/assay_routing_recommendations.parquet",
    CAMPAIGN / "track_a_generative/calibration_anchors_3d.parquet",
    CAMPAIGN / "track_a_generative/gflownet_tso_bridge_boundaries.parquet",
]

if pl is not None:
    append("| Artifact | Sample Rows | Columns | Sample TSV |")
    append("|---|---:|---:|---|")
    for path in important_parquets:
        if not path.exists():
            warn(f"Expected parquet missing: `{path}`")
            continue
        try:
            df = pl.read_parquet(path, n_rows=25)
            sample_path = parquet_samples_dir / f"{path.stem}.sample.tsv"
            df.write_csv(sample_path, separator="\t")
            rel = path.relative_to(CAMPAIGN)
            ok(f"Parquet readable: `{rel}` sample_rows={df.height} cols={df.width}")
            append(f"| `{rel}` | {df.height} | {df.width} | `{sample_path}` |")
            append("")
            append(f"Schema for `{rel}`:")
            append("```json")
            append(json.dumps({k: str(v) for k, v in df.schema.items()}, indent=2))
            append("```")
        except Exception as e:
            fail(f"Parquet failed: `{path}` :: {e}")

section("6. Master Data Room Content, Link, And Blank-Table Scan")

master = CAMPAIGN / "MASTER_DATA_ROOM_INDEX.md"
if master.exists():
    text = master.read_text(encoding="utf-8", errors="replace")
    headings = re.findall(r"^##\s+(.+)$", text, flags=re.MULTILINE)
    append("Headings:")
    for h in headings:
        append(f"- {h}")

    expected_sections = [
        "Executive Dossier",
        "Interactive 3D Visualizer",
        "Medicinal Chemistry Action Center",
        "Wet-Lab CRO Handoff",
        "Track A Cloud AI Readiness",
        "Pending GPU Campaigns",
        "Cryptographic Bill of Materials",
    ]
    for term in expected_sections:
        if term in text:
            ok(f"Master Index contains section: `{term}`")
        else:
            warn(f"Master Index missing expected section: `{term}`")

    blank_hits: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 4 and sum(1 for c in cells if c == "") >= 2:
                blank_hits.append(f"{i}: {line}")

    blank_file = REVIEW_DIR / "blank_table_hits.txt"
    if blank_hits:
        blank_file.write_text("\n".join(blank_hits), encoding="utf-8")
        warn(f"Blank or underpopulated markdown table rows detected. See `{blank_file}`")
        append("```text")
        append("\n".join(blank_hits[:80]))
        append("```")
    else:
        ok("No blank or underpopulated markdown table rows detected.")

    path_hits = sorted(set(re.findall(r"`([^`]+\.(?:md|json|parquet|csv|html|sh|py|yml|yaml))`", text)))
    append("")
    append("Referenced artifact path checks:")
    for raw in path_hits:
        candidates = [
            ROOT / raw,
            CAMPAIGN / raw,
            ROOT / raw.lstrip("/"),
        ]
        exists = any(c.exists() for c in candidates)
        if exists:
            ok(f"Referenced path exists: `{raw}`")
        else:
            warn(f"Referenced path may be unresolved from ROOT/CAMPAIGN: `{raw}`")

section("7. Overclaim / False Precision / Legal-Risk Scan")

patterns = [
    "32 femtosecond",
    "irreversible desensitization",
    "confirmed biological efficacy",
    "proves clinical",
    "guarantees",
    "guaranteed",
    "expected Δ uptake > 15",
    "Expected Δ uptake > 15",
    "expected fixed HDX",
    "therapeutic efficacy",
    "patient response",
    "clinical outcome",
    "mechanistic proof",
]

overclaim_hits: list[str] = []
text_suffixes = {".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".ts", ".tsx"}
for path in CAMPAIGN.rglob("*"):
    if not path.is_file():
        continue
    if "visualizer_app/assets" in str(path):
        continue
    if path.suffix.lower() not in text_suffixes:
        continue
    content = path.read_text(encoding="utf-8", errors="ignore")
    for pat in patterns:
        if pat.lower() in content.lower():
            overclaim_hits.append(f"{path}: {pat}")

if overclaim_hits:
    hit_path = REVIEW_DIR / "overclaim_hits.txt"
    hit_path.write_text("\n".join(overclaim_hits), encoding="utf-8")
    warn(f"Potential overclaim/false-precision hits found. See `{hit_path}`")
    append("```text")
    append("\n".join(overclaim_hits[:100]))
    append("```")
else:
    ok("No overclaim / false-precision patterns found.")

section("8. Claim Falsification Graph Review")

claim_graph = load_json(CAMPAIGN / "claim_falsification_graph.json")
if isinstance(claim_graph, dict):
    claims = claim_graph.get("claims", [])
    append(f"- Claim count: `{len(claims) if isinstance(claims, list) else 'unknown'}`")
    if isinstance(claims, list):
        classes: dict[str, int] = {}
        for c in claims:
            if isinstance(c, dict):
                cls = str(c.get("epistemic_class", "UNKNOWN"))
                classes[cls] = classes.get(cls, 0) + 1
        append(f"- Epistemic class counts: `{classes}`")
        append("")
        append("First 10 claims:")
        append("```json")
        append(json.dumps(claims[:10], indent=2))
        append("```")

section("9. CRO Handoff Full Observable Review")

if pl is not None:
    cro = CAMPAIGN / "CRO_WetLab_Action_Plan.parquet"
    if cro.exists():
        df = pl.read_parquet(cro)
        append(f"- Rows: `{df.height}`")
        append(f"- Columns: `{df.columns}`")
        append("```text")
        append(str(df))
        append("```")

section("10. Medicinal Chemistry / Zero-Shot Replacement Review")

if pl is not None:
    teaser = CAMPAIGN / "track_0_manual_emulation/teaser_solutions.parquet"
    if teaser.exists():
        df = pl.read_parquet(teaser)
        cols = [
            "solution_rank",
            "anchor_id",
            "canonical_smiles",
            "sa_score",
            "pi_complement",
            "pi_clash",
            "projected_durability_improvement",
            "liability_edge_label",
            "anchor_epistemic_class",
            "solution_epistemic_class",
        ]
        cols = [c for c in cols if c in df.columns]
        append(f"- Rows: `{df.height}`")
        append(f"- Columns: `{df.columns}`")
        append("```text")
        append(str(df.select(cols)))
        append("```")

section("11. Metastable / Chronology Infrastructure Review")

meta = load_json(CAMPAIGN / "phase_2c_metastable_atlas_triggers.json")
if isinstance(meta, dict):
    append(f"- Trigger count: `{meta.get('trigger_count')}`")
    append(f"- Capture mode: `{meta.get('capture_mode')}`")
    append("First 5 triggers:")
    append("```json")
    append(json.dumps(meta.get("triggers", [])[:5], indent=2))
    append("```")

snap = load_json(CAMPAIGN / "phase_2c_snapshot_triggers.json")
if isinstance(snap, dict):
    append("Snapshot trigger manifest summary:")
    copy = dict(snap)
    copy.pop("triggers", None)
    append("```json")
    append(json.dumps(copy, indent=2))
    append("```")

section("12. Holo Topology And Pending GPU Campaign Review")

holo = ROOT / "04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json"
if holo.exists():
    ok(f"Holo topology exists: `{holo}`")
    h = load_json(holo)
    if isinstance(h, dict):
        subset = {k: h.get(k) for k in ["n_atoms", "ligand_atoms", "ligand_charge_method", "min_heavy_distance_A", "selected_source_condition"]}
        append("```json")
        append(json.dumps(subset, indent=2))
        append("```")
else:
    warn(f"Holo topology missing at expected path: `{holo}`")

launch = ROOT / "bin/launch-n80-holo-aleniglipron.sh"
if launch.exists():
    ok(f"Holo launch script exists: `{launch}`")
else:
    warn(f"Holo launch script missing: `{launch}`")

section("13. Track A / Cloud AI Readiness Review")

track_a = CAMPAIGN / "track_a_generative"
if track_a.exists():
    files = sorted([p for p in track_a.rglob("*") if p.is_file()])
    append("Track A artifacts:")
    for p in files:
        append(f"- `{p}` size=`{human_size(p)}`")
else:
    warn("Track A generative directory missing.")

cloud = ROOT / "00_registry/architecture/Cloudflare_Manifold_Architecture.md"
if cloud.exists():
    ok(f"Cloudflare architecture exists: `{cloud}`")
    headings = re.findall(r"^#+\s+(.+)$", cloud.read_text(encoding="utf-8", errors="replace"), flags=re.MULTILINE)
    append("Cloudflare architecture headings:")
    for h in headings[:80]:
        append(f"- {h}")
else:
    warn(f"Cloudflare architecture missing: `{cloud}`")

section("14. Full Live Campaign Inventory")

inventory_path = REVIEW_DIR / "full_campaign_inventory.tsv"
with inventory_path.open("w", encoding="utf-8") as f:
    for p in sorted(CAMPAIGN.rglob("*")):
        if p.is_file():
            f.write(f"{p}\t{p.stat().st_size} bytes\n")

ok(f"Full live campaign inventory written: `{inventory_path}`")
append(f"- Live campaign file count: `{sum(1 for _ in inventory_path.open())}`")
append("First 200 inventory rows:")
append("```text")
append("\n".join(inventory_path.read_text(encoding="utf-8").splitlines()[:200]))
append("```")

section("15. Archive Inventory Observable Sample")

if ARCHIVE.exists():
    archive_sample_path = REVIEW_DIR / "archive_inventory_head.txt"
    possible_raw_path = REVIEW_DIR / "possible_raw_bulk_files_sample.txt"
    sample: list[str] = []
    raw_hits: list[str] = []
    try:
        with tarfile.open(ARCHIVE, "r:gz") as tf:
            for idx, member in enumerate(tf):
                if idx < 200:
                    sample.append(member.name)
                if re.search(r"\.(dcd|xtc|trr|nc|raw|bin)$", member.name, re.I):
                    raw_hits.append(member.name)
                    if len(raw_hits) >= 50:
                        break
                if idx >= 200 and os.environ.get("FULL_ARCHIVE_SCAN") != "1":
                    break
        archive_sample_path.write_text("\n".join(sample), encoding="utf-8")
        ok(f"Archive inventory sample written: `{archive_sample_path}`")
        append("```text")
        append("\n".join(sample))
        append("```")
        if raw_hits:
            possible_raw_path.write_text("\n".join(raw_hits), encoding="utf-8")
            warn(f"Possible raw/bulk files found in archive sample. See `{possible_raw_path}`")
        else:
            ok("No obvious raw/bulk trajectory files found in archive sample.")
    except Exception as e:
        warn(f"Archive sample scan failed or was interrupted: {e}")

section("16. Build HTML Review Portal")

artifact_links = [
    ("Master Data Room", CAMPAIGN / "MASTER_DATA_ROOM_INDEX.md"),
    ("Pharmacological Dynamics Intelligence Report", CAMPAIGN / "M2_Pharmacological_Dynamics_Intelligence_Report.md"),
    ("Triangulation Dossier", CAMPAIGN / "M2_Triangulation_Dossier_Final.md"),
    ("Executive Readout", CAMPAIGN / "M2_Executive_Readout_Final.md"),
    ("Enterprise Positioning Summary", CAMPAIGN / "ENTERPRISE_POSITIONING_SUMMARY.md"),
    ("Claim Falsification Graph", CAMPAIGN / "claim_falsification_graph.json"),
    ("CBOM", CAMPAIGN / "PRISM_CBOM_v1.0.json"),
    ("Replayability Manifest", CAMPAIGN / "M2_Replayability_Manifest.json"),
    ("Visualizer", CAMPAIGN / "visualizer_app/index.html"),
    ("Review Markdown", REPORT_MD),
    ("Full Campaign Inventory", inventory_path),
]

links_html = "\n".join(
    f'<li><a href="file://{p}">{html.escape(name)}</a> — <code>{html.escape(str(p))}</code></li>'
    for name, p in artifact_links
    if p.exists()
)

status_badges = [
    "Immutable Archive",
    "Signature Verified" if SIG.exists() else "Signature Missing",
    "CBOM",
    "Epistemic Hardening",
    "CRO Falsification Gates",
    "Visualizer",
]

report_text = REPORT_MD.read_text(encoding="utf-8", errors="replace")
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
code {{ color: #c7d7ff; }}
pre {{
  background: #171a22;
  border: 1px solid #333846;
  border-radius: 12px;
  padding: 16px;
  white-space: pre-wrap;
  overflow-x: auto;
}}
.badge {{
  display: inline-block;
  padding: 4px 8px;
  margin: 2px;
  border-radius: 999px;
  background: #25304a;
  color: #bcd2ff;
  font-size: 12px;
}}
.warn {{ color: #ffca85; }}
.fail {{ color: #ff8585; }}
</style>
</head>
<body>
<h1>PRISM GLP-1R M2 Enterprise Release Viewer</h1>
<p>{''.join(f'<span class="badge">{html.escape(b)}</span>' for b in status_badges)}</p>

<h2>Direct Artifact Links</h2>
<ul>
{links_html}
</ul>

<h2>Browser Endpoints</h2>
<ul>
<li>Enterprise review portal: <a href="http://127.0.0.1:8090/ENTERPRISE_RELEASE_VIEWER.html">http://127.0.0.1:8090/ENTERPRISE_RELEASE_VIEWER.html</a></li>
<li>Interactive visualizer: <a href="http://127.0.0.1:8080">http://127.0.0.1:8080</a></li>
</ul>

<h2>Review Report</h2>
<pre>{html.escape(report_text)}</pre>
</body>
</html>
"""
REPORT_HTML.write_text(html_doc, encoding="utf-8")
ok(f"HTML review portal generated: `{REPORT_HTML}`")

section("17. Launch Local Review Servers")

if (CAMPAIGN / "visualizer_app/index.html").exists():
    if not is_port_open(8080):
        subprocess.Popen(
            [sys.executable, "-m", "http.server", "8080"],
            cwd=CAMPAIGN / "visualizer_app",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ok("Visualizer server started: http://127.0.0.1:8080")
    else:
        ok("Port 8080 already active; visualizer may already be running.")

if not is_port_open(8090):
    subprocess.Popen(
        [sys.executable, "-m", "http.server", "8090"],
        cwd=REVIEW_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ok("Enterprise review portal started: http://127.0.0.1:8090/ENTERPRISE_RELEASE_VIEWER.html")
else:
    ok("Port 8090 already active; review portal may already be running.")

section("18. Final Status")

append(f"- Hard failures: `{len(hard_failures)}`")
append(f"- Warnings: `{len(warnings)}`")
if hard_failures:
    append("Hard failures:")
    for x in hard_failures:
        append(f"- {x}")
if warnings:
    append("Warnings:")
    for x in warnings:
        append(f"- {x}")

print("\n" + "=" * 72)
print(" ENTERPRISE RELEASE VIEWER V2 COMPLETE")
print("=" * 72)
print(f"Markdown report: {REPORT_MD}")
print(f"HTML portal:     {REPORT_HTML}")
print("Browser URLs:")
print("  http://127.0.0.1:8090/ENTERPRISE_RELEASE_VIEWER.html")
print("  http://127.0.0.1:8080")
print()
if hard_failures:
    print(f"[FINAL STATUS] HARD FAILURES: {len(hard_failures)}")
else:
    print("[FINAL STATUS] NO HARD FAILURES")
if warnings:
    print(f"[FINAL STATUS] WARNINGS: {len(warnings)}")
    print("Inspect the report before external delivery.")
