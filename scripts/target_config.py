#!/usr/bin/env python3
"""Target config loader for PRISM-4D unified dossier.

Loads per-target YAML from config/targets/<target_id>.yaml.
Graceful degradation: missing config or empty references list →
substrate-only mode with explicit BLOCKED status on E11/E12/E13/E15/E16
(the reference-anchored enhancements).

Filename → target_id resolution:
  4lpk_clean.topology.spike_events.arrow      → 4lpk
  9m3p_chainA.topology.spike_events.arrow     → 9m3p
  9m3p_clean.topology.spike_events.arrow      → 9m3p
"""
import sys
from pathlib import Path

import yaml


def derive_target_id(arrow_path):
    """Derive a target_id from the Arrow filename.

    Drops `_clean`, `_chainX`, `_v2`, etc. to recover the canonical
    PDB ID stem.
    """
    arrow_basename = Path(arrow_path).name
    stem = arrow_basename.replace(".topology.spike_events.arrow", "")
    # Drop common revision suffixes.
    for suffix in ("_clean", "_v1", "_v2", "_v3"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    # Drop chain suffix (4lpk_chainA → 4lpk).
    if "_chain" in stem:
        stem = stem.split("_chain")[0]
    # Drop merged-seed suffix (4lpk_clean.topology.merged_seeds_42_43_44.arrow
    # collapses to 4lpk_clean.topology.merged after the .arrow strip; we want 4lpk).
    if ".topology.merged" in stem or stem.endswith(".topology"):
        stem = stem.split(".")[0]
        for suffix in ("_clean", "_v1", "_v2", "_v3"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
    return stem.lower()


def load_target_config(target_id, config_root=None):
    """Load `config/targets/<target_id>.yaml`. Returns a dict with the
    schema below regardless of whether the file exists; `found=False`
    means the caller should run in substrate-only mode.

      {
        "found":         bool,
        "config_path":   str | None,
        "target_id":     str,
        "target_name":   str | None,
        "protein_class": str | None,
        "chain_id":      str,
        "domains":       {region_name: set[int]},
        "references":    {pdb_id: {name, het, site, mechanism}},
        "config_raw":    dict | None,
      }
    """
    if config_root is None:
        config_root = Path(__file__).parent.parent / "config" / "targets"

    config_path = Path(config_root) / f"{target_id}.yaml"

    if not config_path.exists():
        return {
            "found": False,
            "config_path": None,
            "target_id": target_id,
            "target_name": None,
            "protein_class": None,
            "chain_id": "A",
            "domains": {},
            "references": {},
            "config_raw": None,
        }

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    domains = {}
    for region_name, region_def in (raw.get("canonical_regions") or {}).items():
        residues = set()
        for r in region_def.get("ranges", []):
            if len(r) == 2:
                residues.update(range(r[0], r[1] + 1))   # INCLUSIVE
        domains[region_name] = residues

    references = {}
    for ref in (raw.get("references") or []):
        pdb_id = ref.get("pdb_id")
        if pdb_id:
            references[pdb_id] = {
                "name":      ref.get("name", pdb_id),
                "het":       ref.get("het", ""),
                "site":      ref.get("site", ""),
                "mechanism": ref.get("mechanism", ""),
            }

    return {
        "found": True,
        "config_path": str(config_path),
        "target_id": target_id,
        "target_name": raw.get("target_name"),
        "protein_class": raw.get("protein_class"),
        "chain_id": raw.get("chain_id", "A"),
        "domains": domains,
        "references": references,
        "config_raw": raw,
    }


def get_substrate_only_blocked_message(target_id):
    """Standard BLOCKED gate message for substrate-only mode."""
    return (
        f"Substrate-only mode: no config/targets/{target_id}.yaml or empty "
        f"references list. Reference-anchored enhancement disabled. "
        f"To enable, create the config file with canonical_regions and "
        f"references."
    )


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "4lpk"
    cfg = load_target_config(tid)
    print(f"target_id: {cfg['target_id']}")
    print(f"found: {cfg['found']}")
    if cfg["found"]:
        print(f"target_name: {cfg['target_name']}")
        print(f"protein_class: {cfg['protein_class']}")
        print(f"chain_id: {cfg['chain_id']}")
        print(f"domains: {list(cfg['domains'].keys())}")
        for k, v in cfg["domains"].items():
            if v:
                print(f"  {k}: {len(v)} residues, range {min(v)}-{max(v)}")
            else:
                print(f"  {k}: 0 residues")
        print(f"references: {list(cfg['references'].keys())}")
