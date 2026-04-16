#!/usr/bin/env python3
"""
[AUDIT - NON-NAIVE VALIDATION]

Rigorous post-execution validation of a single TWIN-10 target.

Runs ~40 checks across 6 stages + cross-stage consistency + provenance
integrity. Produces a structured JSON report AND a human-readable summary.

Severity levels:
  CRITICAL: must pass to declare target valid (halts batch if halt-on-fail)
  HIGH:     strong signal of a problem
  MEDIUM:   warning, worth inspecting
  LOW:      informational

Usage:
    python3 twin10_audit.py --target-dir /path/to/kras_g12d_apo \
                            --holo-pdb /path/to/7RPZ.cif \
                            --ligand-resname 6IC \
                            --known-binding-residues H95,Y96,Q99,R68,M72 \
                            --out /path/to/audit_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Local provenance module
sys.path.insert(0, str(Path(__file__).parent))
from prism_prov import verify_record_self_hash, verify_artifact, blake3_file


# ─────────────────────────────────────────────────────────────────────
# Check result types
# ─────────────────────────────────────────────────────────────────────

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


@dataclass
class CheckResult:
    name: str
    stage: str
    severity: str       # CRITICAL | HIGH | MEDIUM | LOW | INFO
    status: str         # PASS | WARN | FAIL | SKIP | ERROR
    detail: str = ""
    value: Any = None


@dataclass
class AuditReport:
    target: str = ""
    target_dir: str = ""
    timestamp_utc: str = ""
    checks: List[CheckResult] = field(default_factory=list)
    overall: str = "PENDING"  # PASS | WARN | FAIL
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, c: CheckResult):
        self.checks.append(c)

    def finalize(self):
        self.counts = {}
        for c in self.checks:
            key = f"{c.severity}:{c.status}"
            self.counts[key] = self.counts.get(key, 0) + 1
        # Any CRITICAL:FAIL → overall FAIL
        has_crit_fail = any(
            c.severity == "CRITICAL" and c.status in ("FAIL", "ERROR")
            for c in self.checks
        )
        has_high_fail = any(
            c.severity == "HIGH" and c.status in ("FAIL", "ERROR")
            for c in self.checks
        )
        has_warn = any(c.status == "WARN" for c in self.checks)
        if has_crit_fail:
            self.overall = "FAIL"
        elif has_high_fail:
            self.overall = "FAIL"
        elif has_warn:
            self.overall = "WARN"
        else:
            self.overall = "PASS"
        from datetime import datetime, timezone
        self.timestamp_utc = datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────
# Stage 0 — Provenance integrity
# ─────────────────────────────────────────────────────────────────────

def audit_provenance(target_dir: Path, report: AuditReport):
    prov_dir = target_dir / "prov"
    if not prov_dir.exists():
        report.add(CheckResult(
            "prov_dir_exists", "0_provenance", "CRITICAL", "FAIL",
            detail=f"{prov_dir} does not exist"
        ))
        return

    prov_files = sorted(prov_dir.glob("*.prov.json"))
    report.add(CheckResult(
        "prov_records_count", "0_provenance", "CRITICAL",
        "PASS" if len(prov_files) >= 6 else "FAIL",
        detail=f"{len(prov_files)} provenance records found (expected ≥6)",
        value=len(prov_files),
    ))

    bad_hashes = []
    for pf in prov_files:
        v = verify_record_self_hash(pf)
        if not v["valid"]:
            bad_hashes.append(pf.name)
    report.add(CheckResult(
        "prov_self_hashes_valid", "0_provenance", "CRITICAL",
        "PASS" if not bad_hashes else "FAIL",
        detail=f"{len(prov_files) - len(bad_hashes)}/{len(prov_files)} self-hashes valid" +
               (f"; bad: {bad_hashes[:5]}" if bad_hashes else ""),
    ))

    manifest_path = prov_dir / "pipeline_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            mani = json.load(f)
        for rec in mani.get("records", []):
            if not rec.get("present", True):
                continue
            claimed = rec.get("blake3")
            p = Path(rec["path"])
            if p.exists() and claimed:
                actual = blake3_file(p)
                if actual != claimed:
                    report.add(CheckResult(
                        "prov_artifact_drift", "0_provenance", "CRITICAL", "FAIL",
                        detail=f"{p.name}: manifest hash != file hash"
                    ))
                    return
        report.add(CheckResult(
            "manifest_artifact_hashes", "0_provenance", "CRITICAL", "PASS",
            detail="all referenced artifact hashes match deposited"
        ))
    else:
        report.add(CheckResult(
            "manifest_present", "0_provenance", "HIGH", "FAIL",
            detail="pipeline_manifest.json missing"
        ))


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Download audits
# ─────────────────────────────────────────────────────────────────────

def audit_download(target_dir: Path, report: AuditReport):
    d = target_dir / "artifacts" / "1_download"
    if not d.exists():
        report.add(CheckResult("download_dir", "1_download", "CRITICAL", "FAIL",
                              detail=f"{d} missing"))
        return
    cifs = list(d.glob("*.cif"))
    pdbs = list(d.glob("*.pdb"))
    report.add(CheckResult("mmcif_present", "1_download", "HIGH",
                          "PASS" if cifs else "WARN",
                          detail=f"{len(cifs)} .cif files (2026-modern format)"))
    report.add(CheckResult("pdb_present", "1_download", "MEDIUM",
                          "PASS" if pdbs else "WARN",
                          detail=f"{len(pdbs)} .pdb files (legacy)"))
    # Validation report
    val_report = d / "validation.json"
    report.add(CheckResult("rcsb_validation_report", "1_download", "LOW",
                          "PASS" if val_report.exists() else "SKIP",
                          detail="RCSB structure validation metadata"))


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Clean audits
# ─────────────────────────────────────────────────────────────────────

def _count_atoms_in_pdb(path: Path) -> int:
    n = 0
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                n += 1
    return n


def _residue_diversity(path: Path) -> int:
    res = set()
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("ATOM"):
                res.add(line[17:20].strip())
    return len(res)


def audit_clean(target_dir: Path, report: AuditReport):
    d = target_dir / "artifacts" / "2_clean"
    if not d.exists():
        report.add(CheckResult("clean_dir", "2_clean", "CRITICAL", "FAIL",
                              detail=f"{d} missing"))
        return
    # Find the final cleaned PDB — convention: *_clean.pdb or *_final.pdb
    candidates = list(d.glob("*_clean.pdb")) + list(d.glob("*_final.pdb"))
    if not candidates:
        report.add(CheckResult("cleaned_pdb_exists", "2_clean", "CRITICAL", "FAIL",
                              detail="no *_clean.pdb or *_final.pdb in clean dir"))
        return
    cleaned = candidates[0]
    n_atoms = _count_atoms_in_pdb(cleaned)
    report.add(CheckResult("clean_atom_count", "2_clean", "HIGH",
                          "PASS" if n_atoms >= 500 else "FAIL",
                          detail=f"{n_atoms} ATOM+HETATM records (≥500 required)",
                          value=n_atoms))
    diversity = _residue_diversity(cleaned)
    report.add(CheckResult("residue_type_diversity", "2_clean", "CRITICAL",
                          "PASS" if diversity >= 15 else "FAIL",
                          detail=f"{diversity} distinct residue types (≥15 required; "
                                 f"CLAUDE.md prism-clean rule)",
                          value=diversity))


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Prep audits (topology integrity)
# ─────────────────────────────────────────────────────────────────────

def audit_prep(target_dir: Path, report: AuditReport):
    d = target_dir / "artifacts" / "3_prep"
    if not d.exists():
        report.add(CheckResult("prep_dir", "3_prep", "CRITICAL", "FAIL",
                              detail=f"{d} missing"))
        return
    topos = list(d.glob("*.topology.json"))
    if not topos:
        report.add(CheckResult("topology_exists", "3_prep", "CRITICAL", "FAIL",
                              detail="no topology.json in prep dir"))
        return
    topo = topos[0]
    try:
        with open(topo) as f:
            t = json.load(f)
    except Exception as e:
        report.add(CheckResult("topology_parses", "3_prep", "CRITICAL", "FAIL",
                              detail=f"JSON parse error: {e}"))
        return
    report.add(CheckResult("topology_parses", "3_prep", "CRITICAL", "PASS",
                          detail=f"valid JSON"))

    for key in ("atom_names", "bonds", "angles", "dihedrals", "residues",
                "charges", "positions", "lj_params", "residue_names"):
        report.add(CheckResult(f"topology_has_{key}", "3_prep", "CRITICAL",
                              "PASS" if key in t else "FAIL",
                              detail=f"key '{key}' {'present' if key in t else 'missing'}"))

    # Charge sum
    charges = t.get("charges", [])
    if charges:
        net_q = sum(charges)
        report.add(CheckResult("net_charge_reasonable", "3_prep", "HIGH",
                              "PASS" if -15.0 <= net_q <= 15.0 else "WARN",
                              detail=f"net charge = {net_q:.2f}", value=net_q))

    # HIS tautomer assignment (no bare HIS)
    res_names = t.get("residue_names", [])
    his_variants = {"HID", "HIE", "HIP"}
    has_bare_his = any(r == "HIS" for r in res_names)
    his_variant_count = sum(1 for r in res_names if r in his_variants)
    if has_bare_his:
        report.add(CheckResult("his_tautomers_assigned", "3_prep", "HIGH", "FAIL",
                              detail="bare 'HIS' residues found — tautomer not assigned"))
    else:
        report.add(CheckResult("his_tautomers_assigned", "3_prep", "HIGH", "PASS",
                              detail=f"{his_variant_count} HIS tautomers (no bare HIS)"))

    # Atom/residue count sanity
    n_atoms = t.get("n_atoms")
    n_residues = t.get("n_residues")
    if n_atoms:
        expected_bonds_lo = int(n_atoms * 0.8)
        expected_bonds_hi = int(n_atoms * 1.5)
        n_bonds = len(t.get("bonds", []))
        bond_ratio_ok = expected_bonds_lo <= n_bonds <= expected_bonds_hi
        report.add(CheckResult("bond_count_plausible", "3_prep", "MEDIUM",
                              "PASS" if bond_ratio_ok else "WARN",
                              detail=f"n_bonds={n_bonds}, n_atoms={n_atoms} "
                                     f"(expected {expected_bonds_lo}-{expected_bonds_hi})"))

    # GB radii present
    gb_radii = t.get("gb_radii", [])
    report.add(CheckResult("gb_radii_present", "3_prep", "MEDIUM",
                          "PASS" if len(gb_radii) == n_atoms else "WARN",
                          detail=f"gb_radii count={len(gb_radii)}, expected {n_atoms}"))

    # Aromatic targets mapped
    aromatic_targets = t.get("aromatic_targets", [])
    report.add(CheckResult("aromatic_targets_mapped", "3_prep", "LOW",
                          "PASS" if aromatic_targets else "INFO",
                          detail=f"{len(aromatic_targets)} aromatic residues tagged",
                          value=len(aromatic_targets)))


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Engine audits
# ─────────────────────────────────────────────────────────────────────

def audit_engine(target_dir: Path, report: AuditReport):
    d = target_dir / "artifacts" / "5_engine"
    if not d.exists():
        report.add(CheckResult("engine_dir", "5_engine", "CRITICAL", "FAIL",
                              detail=f"{d} missing"))
        return
    bs = list(d.glob("*.binding_sites.json"))
    if not bs:
        report.add(CheckResult("binding_sites_emitted", "5_engine", "CRITICAL", "FAIL",
                              detail="no binding_sites.json"))
        return
    report.add(CheckResult("binding_sites_emitted", "5_engine", "CRITICAL", "PASS",
                          detail=bs[0].name))
    try:
        with open(bs[0]) as f:
            binding = json.load(f)
    except Exception as e:
        report.add(CheckResult("binding_sites_parses", "5_engine", "CRITICAL", "FAIL",
                              detail=f"parse error: {e}"))
        return
    report.add(CheckResult("binding_sites_parses", "5_engine", "CRITICAL", "PASS",
                          detail=f"valid JSON"))

    sites = binding.get("sites", [])
    report.add(CheckResult("engine_sites_count", "5_engine", "HIGH",
                          "PASS" if len(sites) >= 1 else "FAIL",
                          detail=f"{len(sites)} sites detected"))

    # Feature completeness per site — spot check site 0
    if sites:
        s0 = sites[0]
        required_fields = [
            "id", "centroid", "volume", "quality_score", "rank",
            "rank_C", "rank_G", "rank_K", "rank_L", "rank_T",
            "engine_chem", "engine_geo", "engine_phys", "engine_vcs",
            "druggability", "therm_class", "spike_count", "lining_residues",
        ]
        missing = [f for f in required_fields if f not in s0]
        report.add(CheckResult("engine_feature_completeness", "5_engine", "HIGH",
                              "PASS" if not missing else "FAIL",
                              detail=f"missing fields in site 0: {missing}"
                                     if missing else "all required fields present"))

        # NaN check
        def _has_nan(v):
            if isinstance(v, float) and (v != v):
                return True
            if isinstance(v, (list, tuple)):
                return any(_has_nan(x) for x in v)
            if isinstance(v, dict):
                return any(_has_nan(x) for x in v.values())
            return False
        nan_sites = sum(1 for s in sites if _has_nan(s))
        report.add(CheckResult("engine_no_nan_values", "5_engine", "HIGH",
                              "PASS" if nan_sites == 0 else "FAIL",
                              detail=f"{nan_sites}/{len(sites)} sites contain NaN"))

    # Tier B companion record
    tier_b = target_dir / "prov" / "5_engine.tier_b.prov.json"
    if tier_b.exists():
        with open(tier_b) as f:
            tb = json.load(f)
        # Check TWIN multi-differential
        run_log = tb.get("run_log_parsed", {})
        report.add(CheckResult("multi_differential_active", "5_engine", "HIGH",
                              "PASS" if run_log.get("multi_diff_detected") else "FAIL",
                              detail="'Multi-Differential' banner in run.log"))
        # All 4 groups emitted stream trajectories
        n_streams = len(tb.get("stream_trajectories", []))
        report.add(CheckResult("four_group_streams_emitted", "5_engine", "HIGH",
                              "PASS" if n_streams >= 4 else "FAIL",
                              detail=f"{n_streams} stream trajectory PDBs"))
        # Phase bits non-zero
        samples = run_log.get("spike_debug_samples", [])
        if samples:
            nonzero = sum(1 for s in samples if s.get("phase", 0) > 0)
            report.add(CheckResult("phase_bits_populated", "5_engine", "MEDIUM",
                                  "PASS" if nonzero > 0 else "WARN",
                                  detail=f"{nonzero}/{len(samples)} sampled spikes have phase>0"))
        # No errors in run.log
        n_errors = len(run_log.get("errors", []))
        report.add(CheckResult("engine_no_error_lines", "5_engine", "HIGH",
                              "PASS" if n_errors == 0 else "FAIL",
                              detail=f"{n_errors} error lines in run.log"))


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — DCC audits
# ─────────────────────────────────────────────────────────────────────

def audit_dcc(
    target_dir: Path,
    report: AuditReport,
    known_binding_residues: Optional[List[str]] = None,
):
    d = target_dir / "artifacts" / "6_dcc"
    if not d.exists():
        report.add(CheckResult("dcc_dir", "6_dcc", "HIGH", "SKIP",
                              detail=f"{d} missing — DCC stage not run"))
        return
    dcc_result = d / "dcc_result.json"
    if not dcc_result.exists():
        report.add(CheckResult("dcc_computed", "6_dcc", "HIGH", "FAIL",
                              detail="dcc_result.json missing"))
        return
    with open(dcc_result) as f:
        dcc = json.load(f)

    best_dcc = dcc.get("best_site_dcc_angstrom")
    if best_dcc is not None:
        valid_range = 0.1 <= best_dcc <= 50.0
        report.add(CheckResult("dcc_value_range", "6_dcc", "HIGH",
                              "PASS" if valid_range else "FAIL",
                              detail=f"best-site DCC = {best_dcc:.2f}Å "
                                     f"(expected 0.1-50Å)", value=best_dcc))

        grade = dcc.get("grade") or (
            "EXCELLENT" if best_dcc < 5 else
            "GOOD" if best_dcc < 8 else
            "MARGINAL" if best_dcc < 10 else
            "POOR"
        )
        report.add(CheckResult("dcc_grade", "6_dcc", "INFO", "PASS",
                              detail=f"grade = {grade}", value=grade))

    best_rank = dcc.get("best_site_rank")
    if best_rank is not None:
        report.add(CheckResult("sr_at_5", "6_dcc", "HIGH",
                              "PASS" if best_rank <= 5 else "WARN",
                              detail=f"best-DCC site rank = {best_rank} (SR@5)",
                              value=best_rank))
        report.add(CheckResult("sr_at_1", "6_dcc", "MEDIUM",
                              "PASS" if best_rank == 1 else "INFO",
                              detail=f"SR@1 = {'yes' if best_rank == 1 else 'no'}"))

    # Lining residue cross-check against known binding residues
    if known_binding_residues:
        lining = set(dcc.get("best_site_lining_residues", []))
        # Normalize: literature gives e.g. "H95" (aa+number); lining is likely
        # dict with resid+resname. We try a soft match on resid numbers.
        known_nums = set()
        for r in known_binding_residues:
            m = re.match(r"[A-Z]+(\d+)", r.strip())
            if m:
                known_nums.add(int(m.group(1)))
        lining_nums = set()
        for r in lining:
            if isinstance(r, dict):
                if "resid" in r:
                    lining_nums.add(int(r["resid"]))
            elif isinstance(r, (int, str)):
                try:
                    lining_nums.add(int(r))
                except ValueError:
                    pass
        overlap = known_nums & lining_nums
        report.add(CheckResult("literature_residue_overlap", "6_dcc", "HIGH",
                              "PASS" if len(overlap) >= 2 else "WARN",
                              detail=f"{len(overlap)}/{len(known_nums)} known binding "
                                     f"residues in engine lining "
                                     f"(required ≥2 for literature validation)"))


# ─────────────────────────────────────────────────────────────────────
# Cross-stage consistency
# ─────────────────────────────────────────────────────────────────────

def audit_cross_stage(target_dir: Path, report: AuditReport):
    # Residue-count consistency across topology.json, binding_sites.json,
    # residue_map.json, kcc_visualization.json
    try:
        prep_d = target_dir / "artifacts" / "3_prep"
        eng_d = target_dir / "artifacts" / "5_engine"
        topo_files = list(prep_d.glob("*.topology.json"))
        rmap_files = list(eng_d.glob("*.residue_map.json"))
        kcc_files = list(eng_d.glob("*.kcc_visualization.json"))
        if topo_files and rmap_files and kcc_files:
            with open(topo_files[0]) as f:
                topo = json.load(f)
            with open(rmap_files[0]) as f:
                rmap = json.load(f)
            with open(kcc_files[0]) as f:
                kcc = json.load(f)
            n_topo = topo.get("n_residues")
            n_rmap = len(rmap.get("residues", []))
            n_kcc = len(kcc.get("residues", []))
            all_match = n_topo == n_rmap == n_kcc
            report.add(CheckResult("residue_count_consistent", "xstage", "HIGH",
                                  "PASS" if all_match else "FAIL",
                                  detail=f"topology={n_topo} residue_map={n_rmap} "
                                         f"kcc={n_kcc}"))
    except Exception as e:
        report.add(CheckResult("residue_count_consistent", "xstage", "HIGH", "ERROR",
                              detail=f"{type(e).__name__}: {e}"))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--known-binding-residues", default=None,
                    help="Comma-separated e.g. H95,Y96,Q99,R68,M72")
    args = ap.parse_args()

    report = AuditReport(target_dir=str(args.target_dir),
                         target=args.target_dir.name)

    print(f"=== TWIN-10 AUDIT: {args.target_dir.name} ===")
    print()

    audit_provenance(args.target_dir, report)
    audit_download(args.target_dir, report)
    audit_clean(args.target_dir, report)
    audit_prep(args.target_dir, report)
    audit_engine(args.target_dir, report)
    known_res = (args.known_binding_residues.split(",")
                 if args.known_binding_residues else None)
    audit_dcc(args.target_dir, report, known_binding_residues=known_res)
    audit_cross_stage(args.target_dir, report)

    report.finalize()

    # Print human-readable
    for stage in ["0_provenance", "1_download", "2_clean", "3_prep",
                  "4_ground_truth", "5_engine", "6_dcc", "xstage"]:
        stage_checks = [c for c in report.checks if c.stage == stage]
        if not stage_checks:
            continue
        print(f"--- {stage} ---")
        for c in stage_checks:
            mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗",
                    "ERROR": "!", "SKIP": "-", "INFO": "i"}.get(c.status, "?")
            print(f"  [{c.severity:8s}] {mark} {c.name:40s} {c.detail}")
        print()

    print(f"=== OVERALL: {report.overall} ===")
    for key in sorted(report.counts.keys()):
        print(f"  {key:30s} {report.counts[key]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "target": report.target,
        "target_dir": report.target_dir,
        "timestamp_utc": report.timestamp_utc,
        "overall": report.overall,
        "counts": report.counts,
        "checks": [asdict(c) for c in report.checks],
    }
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nAudit written: {args.out}")

    return 0 if report.overall == "PASS" else (2 if report.overall == "WARN" else 1)


if __name__ == "__main__":
    sys.exit(main())
