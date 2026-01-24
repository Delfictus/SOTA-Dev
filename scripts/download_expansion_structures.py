#!/usr/bin/env python3
"""
PRISM-Zero v3.1.1 Structure Expansion Downloader

Downloads additional PDB structures for calibration set expansion.
Fetches kinase, chaperone, protease, and enzyme structures with known cryptic sites.

IMPORTANT: Does NOT download any holdout structures:
- 6W41 (SARS-CoV-2 RBD cryptic epitope)
- 1YCR (MDM2 p53-binding cryptic pocket)
- 2W3L (BCL-2 BH3-binding groove)

Usage:
    python3 download_expansion_structures.py [--output data/calibration/expansion]
"""

import os
import sys
import argparse
import subprocess
import ssl
import urllib.request
import urllib.error
from pathlib import Path

# Create SSL context that doesn't verify certificates (for systems with SSL issues)
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# ============================================================================
# EXPANSION STRUCTURES
# ============================================================================

# Structures to download with their metadata
# Format: (pdb_id, family, difficulty, description, cryptic_site_type)
EXPANSION_STRUCTURES = [
    # Kinase family - allosteric and DFG-out pockets
    ("3GCS", "kinase", "hard", "p38alpha MAP kinase apo form - allosteric back pocket closed", "allosteric"),
    ("1A9U", "kinase", "medium", "p38alpha MAP kinase with SB203580 - back pocket open", "allosteric"),
    ("1CM8", "kinase", "medium", "CDK2 apo form - C-helix out", "allosteric"),
    ("1HCK", "kinase", "easy", "CDK2 with ATP - active conformation", "allosteric"),

    # Chaperone family - ATP lid and client binding
    ("1YET", "chaperone", "hard", "HSP90 N-terminal domain apo - ATP lid closed", "cryptic_lid"),
    ("2VCI", "chaperone", "medium", "HSP90 with geldanamycin - lid partially open", "cryptic_lid"),
    ("1AM1", "chaperone", "hard", "HSP70 NBD apo - interdomain interface closed", "allosteric"),

    # Protease family - flap dynamics
    ("1HHP", "protease", "hard", "HIV-1 protease apo - flaps closed", "flap_cryptic"),
    ("1HVR", "protease", "medium", "HIV-1 protease with inhibitor - flaps open", "flap_cryptic"),
    ("2PC0", "protease", "expert", "HIV-1 protease multi-drug resistant - altered dynamics", "flap_cryptic"),

    # Enzyme family - omega-loop and active site access
    ("1BTL", "enzyme", "medium", "TEM-1 beta-lactamase apo - omega loop closed", "omega_loop"),
    ("1TEM", "enzyme", "easy", "TEM-1 beta-lactamase with inhibitor", "omega_loop"),

    # Nuclear receptor family - coactivator groove
    ("1HG4", "nuclear_receptor", "hard", "Androgen receptor LBD apo - AF2 groove closed", "coactivator"),
    ("2AM9", "nuclear_receptor", "medium", "Androgen receptor with agonist - groove open", "coactivator"),

    # Additional expert-tier structures for stress testing
    ("3P0G", "kinase", "expert", "BRAF V600E mutant - DFG-out cryptic pocket", "dfg_out"),
    ("4MNE", "kinase", "expert", "BRAF with vemurafenib - induced pocket", "induced_fit"),
]

# HOLDOUT STRUCTURES - NEVER DOWNLOAD THESE
HOLDOUT_PDB_IDS = {"6W41", "1YCR", "2W3L"}


def download_pdb(pdb_id: str, output_dir: Path) -> bool:
    """Download a PDB file from RCSB using multiple fallback methods."""

    # Safety check
    if pdb_id.upper() in HOLDOUT_PDB_IDS:
        print(f"  [BLOCKED] {pdb_id} is a holdout structure - SKIPPING")
        return False

    output_path = output_dir / f"{pdb_id.upper()}.pdb"

    if output_path.exists():
        print(f"  [EXISTS] {pdb_id} already downloaded")
        return True

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"

    # Try multiple download methods
    success = False

    # Method 1: Try wget first (most reliable)
    if not success:
        try:
            print(f"  [DOWNLOADING] {pdb_id} via wget...")
            result = subprocess.run(
                ["wget", "-q", "--no-check-certificate", "-O", str(output_path), url],
                capture_output=True,
                timeout=60
            )
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                success = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Method 2: Try curl
    if not success:
        try:
            print(f"  [DOWNLOADING] {pdb_id} via curl...")
            result = subprocess.run(
                ["curl", "-s", "-k", "-o", str(output_path), url],
                capture_output=True,
                timeout=60
            )
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                success = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Method 3: Try urllib with SSL context
    if not success:
        try:
            print(f"  [DOWNLOADING] {pdb_id} via urllib (no SSL verify)...")
            with urllib.request.urlopen(url, context=SSL_CONTEXT, timeout=60) as response:
                data = response.read()
                with open(output_path, 'wb') as f:
                    f.write(data)
            if output_path.exists() and output_path.stat().st_size > 1000:
                success = True
        except Exception as e:
            if output_path.exists():
                output_path.unlink()

    # Verify result
    if success and output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  [SUCCESS] {pdb_id} saved ({output_path.stat().st_size:,} bytes)")
        return True
    else:
        if output_path.exists():
            output_path.unlink()
        print(f"  [FAILED] {pdb_id} - all download methods failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download expansion structures for PRISM-Zero calibration"
    )
    parser.add_argument(
        "--output",
        default="data/calibration/expansion",
        help="Output directory for PDB files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    args = parser.parse_args()

    # Resolve path relative to script location or cwd
    script_dir = Path(__file__).parent.parent  # Go up from scripts/
    output_dir = script_dir / args.output

    print("=" * 70)
    print("PRISM-Zero v3.1.1 Structure Expansion Downloader")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Structures to download: {len(EXPANSION_STRUCTURES)}")
    print(f"Holdout structures (blocked): {HOLDOUT_PDB_IDS}")
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No files will be downloaded]")
        print()

    # Group structures by family for nice display
    by_family = {}
    for pdb_id, family, difficulty, desc, site_type in EXPANSION_STRUCTURES:
        if family not in by_family:
            by_family[family] = []
        by_family[family].append((pdb_id, difficulty, desc, site_type))

    print("Structures by family:")
    print("-" * 70)
    for family, structures in sorted(by_family.items()):
        print(f"\n{family.upper()} ({len(structures)} structures):")
        for pdb_id, difficulty, desc, site_type in structures:
            print(f"  - {pdb_id}: [{difficulty}] {desc}")
    print()

    if args.dry_run:
        print("Run without --dry-run to download these structures.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download structures
    print("Downloading structures...")
    print("-" * 70)

    success_count = 0
    failed = []

    for pdb_id, family, difficulty, desc, site_type in EXPANSION_STRUCTURES:
        if download_pdb(pdb_id, output_dir):
            success_count += 1
        else:
            failed.append(pdb_id)

    print()
    print("=" * 70)
    print(f"Download complete: {success_count}/{len(EXPANSION_STRUCTURES)} successful")

    if failed:
        print(f"Failed downloads: {', '.join(failed)}")
        print("You may need to download these manually from https://www.rcsb.org/")

    print()
    print("Next steps:")
    print("1. Run: python3 scripts/generate_expansion_manifest.py")
    print("2. Review: data/manifests/expanded_calibration.json")
    print("3. Train: ./target/release/prism-train-neuro --manifest data/manifests/expanded_calibration.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
