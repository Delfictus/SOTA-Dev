#!/usr/bin/env python3
"""
Generate batched nhs_rt_full run commands for CryptoBench benchmark.

Groups structures into batches of 5 and generates shell scripts for each batch.
Each structure runs with:
    nhs_rt_full -t <topo>.json -o <results_dir> \
        --fast --hysteresis --multi-stream 8 \
        --spike-percentile 95 --rt-clustering -v

Usage:
    python 03_generate_run_commands.py [--batch-size 5] [--test-only]
"""

import json
import sys
import os
import stat
from pathlib import Path

BENCH_DIR = Path(__file__).parent.resolve()
PRISM_ROOT = BENCH_DIR.parent.parent
TOPOLOGIES_DIR = BENCH_DIR / "topologies"
RESULTS_DIR = BENCH_DIR / "results"
BATCHES_DIR = BENCH_DIR / "batches"
DATA_DIR = BENCH_DIR / "data"

# The exact command specified by the user
NHS_RT_FULL = str(PRISM_ROOT / "target" / "release" / "nhs_rt_full")
RUN_FLAGS = "--fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate batched nhs_rt_full commands for CryptoBench"
    )
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Structures per batch (default: 5)")
    parser.add_argument("--test-only", action="store_true",
                       help="Only generate for test-set structures")
    args = parser.parse_args()

    # Load manifest
    manifest_path = BENCH_DIR / "run_manifest.txt"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run 02_prepare_topologies.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        all_ids = [line.strip() for line in f if line.strip()]

    # Filter to test set if requested
    if args.test_only:
        folds_path = DATA_DIR / "folds.json"
        if folds_path.exists():
            with open(folds_path) as f:
                folds = json.load(f)
            test_ids = set(folds.get("test", []))
            all_ids = [pid for pid in all_ids if pid in test_ids]
            print(f"Test set: {len(all_ids)} structures")
        else:
            print("WARNING: folds.json not found, using full manifest")

    # Filter to only those with existing topology files
    ready_ids = []
    missing = []
    for apo_id in all_ids:
        topo = TOPOLOGIES_DIR / f"{apo_id}.topology.json"
        if topo.exists():
            ready_ids.append(apo_id)
        else:
            missing.append(apo_id)

    if missing:
        print(f"WARNING: {len(missing)} structures missing topologies, skipping")

    print(f"Ready to run: {len(ready_ids)} structures")
    print(f"Batch size: {args.batch_size}")

    # Create batch directories
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate batches
    batches = []
    for i in range(0, len(ready_ids), args.batch_size):
        batch = ready_ids[i:i + args.batch_size]
        batches.append(batch)

    print(f"Total batches: {len(batches)}")
    print()

    # Write per-batch scripts
    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        script_path = BATCHES_DIR / f"batch_{batch_num:03d}.sh"

        lines = [
            "#!/usr/bin/env bash",
            f"# CryptoBench Batch {batch_num}/{len(batches)} — {len(batch)} structures",
            f"# Structures: {', '.join(batch)}",
            "set -euo pipefail",
            "",
            f'BENCH_DIR="{BENCH_DIR}"',
            f'TOPO_DIR="{TOPOLOGIES_DIR}"',
            f'RESULTS_DIR="{RESULTS_DIR}"',
            f'NHS="{NHS_RT_FULL}"',
            "",
            f"echo '=========================================='",
            f"echo '  CryptoBench Batch {batch_num}/{len(batches)}'",
            f"echo '=========================================='",
            f"echo 'Structures: {' '.join(batch)}'",
            "echo ''",
            "",
        ]

        for j, apo_id in enumerate(batch, 1):
            topo_file = f"${{TOPO_DIR}}/{apo_id}.topology.json"
            out_dir = f"${{RESULTS_DIR}}/{apo_id}"
            lines.extend([
                f"# --- [{j}/{len(batch)}] {apo_id} ---",
                f'echo "[{j}/{len(batch)}] Running {apo_id}..."',
                f'mkdir -p "{out_dir}"',
                f'START=$(date +%s)',
                f'"$NHS" -t "{topo_file}" -o "{out_dir}" {RUN_FLAGS} 2>&1 | tee "{out_dir}/run.log"',
                f'EXIT_CODE=${{PIPESTATUS[0]}}',
                f'END=$(date +%s)',
                f'ELAPSED=$((END - START))',
                f'echo "  {apo_id}: exit=${{EXIT_CODE}}, time=${{ELAPSED}}s"',
                f'echo "{apo_id},${{EXIT_CODE}},${{ELAPSED}}" >> "${{RESULTS_DIR}}/batch_{batch_num:03d}_timing.csv"',
                "",
            ])

        lines.extend([
            f"echo ''",
            f"echo 'Batch {batch_num} complete.'",
            f"echo ''",
        ])

        with open(script_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)

    # Write master run-all script
    master_path = BENCH_DIR / "run_all_batches.sh"
    master_lines = [
        "#!/usr/bin/env bash",
        "# CryptoBench Master Runner — All batches sequentially",
        f"# Total: {len(ready_ids)} structures in {len(batches)} batches of {args.batch_size}",
        "set -euo pipefail",
        "",
        f'BENCH_DIR="{BENCH_DIR}"',
        f'BATCHES_DIR="{BATCHES_DIR}"',
        f'RESULTS_DIR="{RESULTS_DIR}"',
        "",
        "echo '============================================='",
        f"echo '  CryptoBench Full Benchmark Run'",
        f"echo '  {len(ready_ids)} structures / {len(batches)} batches'",
        "echo '============================================='",
        "echo ''",
        "",
        f"START_ALL=$(date +%s)",
        "",
    ]

    for batch_idx in range(len(batches)):
        batch_num = batch_idx + 1
        master_lines.extend([
            f'echo "=== Batch {batch_num}/{len(batches)} ==="',
            f'bash "${{BATCHES_DIR}}/batch_{batch_num:03d}.sh"',
            "",
        ])

    master_lines.extend([
        "END_ALL=$(date +%s)",
        "TOTAL_TIME=$((END_ALL - START_ALL))",
        'echo "============================================="',
        f'echo "  CryptoBench complete: {len(ready_ids)} structures"',
        'echo "  Total time: ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m $(( TOTAL_TIME % 60 ))s)"',
        'echo "============================================="',
    ])

    with open(master_path, 'w') as f:
        f.write('\n'.join(master_lines) + '\n')
    os.chmod(master_path, os.stat(master_path).st_mode | stat.S_IEXEC)

    # Write individual batch commands for copy-paste
    commands_path = BENCH_DIR / "BATCH_COMMANDS.md"
    with open(commands_path, 'w') as f:
        f.write(f"# CryptoBench Batch Commands\n\n")
        f.write(f"Total: {len(ready_ids)} structures in {len(batches)} batches of {args.batch_size}\n\n")
        f.write(f"## Quick Start\n\n")
        f.write(f"```bash\n")
        f.write(f"# Run everything:\n")
        f.write(f"bash {master_path}\n")
        f.write(f"\n# Or run individual batches:\n")
        for batch_idx in range(min(3, len(batches))):
            batch_num = batch_idx + 1
            f.write(f"bash {BATCHES_DIR}/batch_{batch_num:03d}.sh\n")
        if len(batches) > 3:
            f.write(f"# ... ({len(batches) - 3} more batches)\n")
        f.write(f"```\n\n")

        f.write(f"## Individual Structure Commands\n\n")
        f.write(f"Each structure runs:\n```\n")
        f.write(f"nhs_rt_full -t <topo>.json -o <results_dir> {RUN_FLAGS}\n")
        f.write(f"```\n\n")

        f.write(f"## All Batches\n\n")
        for batch_idx, batch in enumerate(batches):
            batch_num = batch_idx + 1
            f.write(f"### Batch {batch_num} ({', '.join(batch)})\n")
            f.write(f"```bash\n")
            f.write(f"bash {BATCHES_DIR}/batch_{batch_num:03d}.sh\n")
            f.write(f"```\n\n")
            f.write(f"Or manually:\n```bash\n")
            for apo_id in batch:
                topo = f"{TOPOLOGIES_DIR}/{apo_id}.topology.json"
                out = f"{RESULTS_DIR}/{apo_id}"
                f.write(f"mkdir -p {out} && {NHS_RT_FULL} -t {topo} -o {out} {RUN_FLAGS}\n")
            f.write(f"```\n\n")

    print()
    print("=" * 60)
    print("  BATCH COMMANDS GENERATED")
    print("=" * 60)
    print(f"  Structures: {len(ready_ids)}")
    print(f"  Batches: {len(batches)} (x{args.batch_size} each)")
    print(f"  Batch scripts: {BATCHES_DIR}/batch_NNN.sh")
    print(f"  Master script: {master_path}")
    print(f"  Commands doc: {commands_path}")
    print()
    print("  To run everything:")
    print(f"    bash {master_path}")
    print()
    print("  To run a single batch:")
    print(f"    bash {BATCHES_DIR}/batch_001.sh")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
