#!/usr/bin/env python3
"""Build target_config.json + serial launcher for the M1 strict-DCC panel.

Read-only wrt pipeline state. Writes target_config.json per target into a
dedicated panel output directory, and emits a launcher shell script.
Does not run the engine — just prepares the materials.
"""
from __future__ import annotations
import json
from pathlib import Path

BLIND_MANIFEST = Path("/mnt/storage/prism-outputs/ml/blind_validation/blind_100_manifest.json")
# Panel output root redirected 2026-04-18 to avoid /mnt/storage ENOSPC during
# in-flight R2 offload. See scripts/quarantine/PANEL_OUTPUT_ROOT.md for the
# full script inventory. The old root remains the READ path for already-
# completed targets; this is the WRITE path for NEW panel runs.
PANEL_BASE = Path.home() / "prism-working" / "m1-strict-dcc-panel"
LAUNCHER = Path("/tmp/m1_strict_panel_launcher.sh")
MANIFEST_OUT = Path("/tmp/m1_strict_panel_manifest.json")
REPO = "/home/diddy/Desktop/Prism4D-bio"

# Pre-declared 11 fresh targets from blind_validation_100 (lowest pRMSD with ligand).
FRESH_IDS = ["2nvp", "1xhx", "2akr", "2e3k", "3bjp", "6tyo", "7se6", "1k47", "5yj2", "3umi", "3bl7"]


def main() -> None:
    blind = json.loads(BLIND_MANIFEST.read_text())
    cb = {t["apo_id"].lower(): t for t in blind["cryptobench_targets"]}

    panel = []
    # Existing 4 — already verified
    for apo, holo, lig, target_dir in [
        ("6yhr", "8pfo", "YHC", "wrn_apo"),
        ("3re2", "7uj4", "OQ4", "menin_apo"),
        ("4qy4", "5dkc", "5BW", "smarca2_brd_apo"),
        ("3p1a", "8d6e", "QGI", "pkmyt1_apo"),
    ]:
        panel.append({
            "target": target_dir, "apo_pdb": apo, "holo_pdb": holo, "ligand": lig,
            "blocker": "none", "source": "TWIN-10 done", "runtime_estimate_min": 0,
            "target_dir": f"/mnt/storage/prism-outputs/twin-10-patent/{target_dir}",
        })

    # 3 TWIN-10 recoveries
    panel.append({
        "target": "kras_g12d_apo", "apo_pdb": "7f0w", "holo_pdb": "7rpz", "ligand": "6IC",
        "blocker": "engine_never_ran (fix landed: 63dd53a9 type-guard)",
        "source": "TWIN-10 recovery", "runtime_estimate_min": 45,
        "target_dir": "/mnt/storage/prism-outputs/twin-10-patent/kras_g12d_apo",
    })
    panel.append({
        "target": "usp1_apo", "apo_pdb": "7ay0", "holo_pdb": "9di1", "ligand": "A1A4Y",
        "blocker": "9DI1 not downloaded — re-run stage 1",
        "source": "TWIN-10 recovery", "runtime_estimate_min": 5,
        "target_dir": "/mnt/storage/prism-outputs/twin-10-patent/usp1_apo",
    })
    panel.append({
        "target": "polq_apo", "apo_pdb": "6xbu", "holo_pdb": "8e24", "ligand": "auto-pick",
        "blocker": "stage-4 superposition fail: too few common Cα",
        "source": "TWIN-10 recovery", "runtime_estimate_min": 10,
        "target_dir": "/mnt/storage/prism-outputs/twin-10-patent/polq_apo",
    })

    # 11 fresh from CryptoBench paired set (pRMSD < 2.1 Å)
    for apo_id in FRESH_IDS:
        rec = cb[apo_id]
        panel.append({
            "target": f"m1_{apo_id}", "apo_pdb": apo_id, "holo_pdb": rec["holo_pdb"],
            "ligand": rec["ligand"], "blocker": "none",
            "source": "blind_validation_100 (CryptoBench)",
            "runtime_estimate_min": 60,
            "target_dir": str(PANEL_BASE / f"m1_{apo_id}"),
            "apo_chain": rec.get("apo_chain", "A"),
            "holo_chain": rec.get("holo_chain", "A"),
            "pRMSD": rec.get("pRMSD"),
        })

    # Emit target_config.json for the 11 fresh targets
    PANEL_BASE.mkdir(parents=True, exist_ok=True)
    for entry in panel:
        if entry["source"] != "blind_validation_100 (CryptoBench)":
            continue
        tdir = Path(entry["target_dir"])
        tdir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "target": entry["target"],
            "pdb_id": entry["apo_pdb"].upper(),
            "chain": entry.get("apo_chain", "A"),
            "paired_holo_pdb_id": entry["holo_pdb"].upper(),
            "paired_holo_ligand_resname": entry["ligand"],
            "paired_holo_chain": entry.get("holo_chain", "A"),
            "source": "blind_validation_100",
            "pRMSD_angstrom": entry.get("pRMSD"),
        }
        (tdir / "target_config.json").write_text(json.dumps(cfg, indent=2))

    # Emit manifest
    total_runtime = sum(e["runtime_estimate_min"] for e in panel)
    manifest = {
        "n_targets": len(panel),
        "tier": "A (strict DCC)",
        "total_runtime_estimate_min": total_runtime,
        "panel": panel,
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))

    # Print manifest table
    print(f"{'target':<22} {'apo':<6} {'holo':<6} {'ligand':<8} {'ETA_min':>7}  blocker")
    print(f"{'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*7}  {'-'*40}")
    for e in panel:
        print(f"{e['target']:<22} {e['apo_pdb']:<6} {e['holo_pdb']:<6} {e['ligand']:<8} "
              f"{e['runtime_estimate_min']:>7}  {e['blocker']}")
    print()
    print(f"n_targets = {len(panel)} (target: 18)")
    print(f"total_runtime_estimate_min = {total_runtime}  (~{total_runtime/60:.1f} h)")
    print(f"manifest: {MANIFEST_OUT}")

    # Emit launcher for the 11 fresh + 3 recoveries
    launch_lines = [
        "#!/bin/bash",
        "# M1 strict-DCC panel launcher. Serial runs (GPU-bound). Background-safe.",
        "set -uo pipefail",
        f"REPO={REPO}",
        f"RUNSTAGES=$REPO/scripts/quarantine/run_stages.py",
        "LOG=/tmp/m1_strict_panel_$(date +%Y%m%d_%H%M%S).log",
        'echo "═══ M1 STRICT-DCC PANEL START $(date) ═══" | tee -a "$LOG"',
    ]
    # USP1 recovery
    launch_lines.append("""
# USP1 recovery: re-run stages 1+4 (9DI1 missing)
for s in 1_download 4_ground_truth 7_evaluation; do
  echo "=== usp1_apo $s ===" | tee -a "$LOG"
  python3 $RUNSTAGES --target-config /mnt/storage/prism-outputs/twin-10-patent/usp1_apo/target_config.json \\
      --stage $s --target-dir /mnt/storage/prism-outputs/twin-10-patent/usp1_apo 2>&1 | tee -a "$LOG"
done
""")
    # POLQ recovery
    launch_lines.append("""
# POLQ recovery: re-run stage 4 (may need resname fix)
for s in 4_ground_truth 7_evaluation; do
  echo "=== polq_apo $s ===" | tee -a "$LOG"
  python3 $RUNSTAGES --target-config /mnt/storage/prism-outputs/twin-10-patent/polq_apo/target_config.json \\
      --stage $s --target-dir /mnt/storage/prism-outputs/twin-10-patent/polq_apo 2>&1 | tee -a "$LOG"
done
""")
    # KRAS full re-run
    launch_lines.append("""
# KRAS full re-run (engine never executed)
echo "=== kras_g12d_apo FULL ===" | tee -a "$LOG"
python3 $RUNSTAGES --target-config /mnt/storage/prism-outputs/twin-10-patent/kras_g12d_apo/target_config.json \\
    --stage all --target-dir /mnt/storage/prism-outputs/twin-10-patent/kras_g12d_apo --no-cupti 2>&1 | tee -a "$LOG"
""")
    # 11 fresh targets
    for apo_id in FRESH_IDS:
        launch_lines.append(f"""
echo "=== m1_{apo_id} FULL $(date) ===" | tee -a "$LOG"
python3 $RUNSTAGES --target-config /mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_{apo_id}/target_config.json \\
    --stage all --target-dir /mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_{apo_id} --no-cupti 2>&1 | tee -a "$LOG"
""")
    launch_lines.append('echo "═══ M1 STRICT-DCC PANEL DONE $(date) ═══" | tee -a "$LOG"')
    LAUNCHER.write_text("\n".join(launch_lines))
    LAUNCHER.chmod(0o755)
    print(f"launcher: {LAUNCHER}")


if __name__ == "__main__":
    main()
