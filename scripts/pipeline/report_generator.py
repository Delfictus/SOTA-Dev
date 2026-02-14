"""Publication-quality report generator for PRISM-4D pipeline.

Produces per-compound hit reports and campaign-level summaries as markdown
files, rendered from Jinja2 templates.

Usage::

    gen = ReportGenerator(output_dir="/tmp/campaign", target_name="KRAS_G12C")
    gen.add_candidate(filtered_candidate, fep_result=..., ensemble=..., ...)
    gen.set_pocket_data(solvent_result, water_map, pocket_dynamics)
    gen.set_audit_trail(audit_trail)
    gen.generate_all()
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None  # type: ignore[assignment,misc]
    FileSystemLoader = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "report_templates"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class _FallbackRenderer:
    """Minimal fallback when Jinja2 is not installed."""

    @staticmethod
    def render(template_name: str, context: Dict[str, Any]) -> str:
        lines = [f"# Report ({template_name})"]
        for key, val in context.items():
            if isinstance(val, (str, int, float, bool)):
                lines.append(f"**{key}:** {val}")
        return "\n".join(lines)


class ReportGenerator:
    """Generates publication-quality markdown reports."""

    def __init__(
        self,
        output_dir: str,
        target_name: str = "UNKNOWN",
        pdb_id: str = "UNKNOWN",
        project_name: str = "PRISM-4D Campaign",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.target_name = target_name
        self.pdb_id = pdb_id
        self.project_name = project_name
        self._candidates: List[Dict[str, Any]] = []
        self._pocket_data: Dict[str, Any] = {}
        self._audit_data: Dict[str, Any] = {}
        self._stages: List[Dict[str, Any]] = []
        self._n_generated: int = 0
        self._n_filtered: int = 0

        if Environment is not None and _TEMPLATE_DIR.is_dir():
            env = Environment(
                loader=FileSystemLoader(str(_TEMPLATE_DIR)),
                keep_trailing_newline=True,
            )
            self._hit_template = env.get_template("hit_report.md.j2")
            self._campaign_template = env.get_template("campaign_report.md.j2")
            self._use_jinja = True
        else:
            self._use_jinja = False
            logger.warning(
                "Jinja2 not available or templates not found; using fallback renderer"
            )

    # ── Data injection ────────────────────────────────────────────────

    def add_candidate(
        self,
        candidate_dict: Dict[str, Any],
        fep_result: Optional[Dict[str, Any]] = None,
        ensemble_result: Optional[Dict[str, Any]] = None,
        ie_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a compound candidate with optional scoring results."""
        entry: Dict[str, Any] = {
            "compound_id": candidate_dict.get(
                "compound_id",
                candidate_dict.get("molecule", {}).get("smiles", "UNK")[:20],
            ),
            "smiles": candidate_dict.get("molecule", {}).get("smiles", ""),
            "source": candidate_dict.get("molecule", {}).get("source", ""),
            "qed_score": candidate_dict.get("qed_score", 0.0),
            "sa_score": candidate_dict.get("sa_score", 0.0),
            "lipinski_violations": candidate_dict.get("lipinski_violations", 0),
            "pains_alerts": candidate_dict.get("pains_alerts", []),
            "tanimoto_to_nearest_known": candidate_dict.get(
                "tanimoto_to_nearest_known", 0.0
            ),
            "nearest_known_cid": candidate_dict.get("nearest_known_cid", ""),
            "cluster_id": candidate_dict.get("cluster_id", -1),
            "pharmacophore_match_score": candidate_dict.get("molecule", {}).get(
                "pharmacophore_match_score", 0.0
            ),
        }

        # Ensemble data
        if ensemble_result:
            entry["ensemble_dg_mean"] = ensemble_result.get("delta_g_mean", 0.0)
            entry["ensemble_dg_std"] = ensemble_result.get("delta_g_std", 0.0)
            entry["ensemble_n_snapshots"] = ensemble_result.get("n_snapshots", 0)
            entry["ensemble_dg"] = (
                f"{entry['ensemble_dg_mean']:.2f} +/- {entry['ensemble_dg_std']:.2f}"
            )
            entry["has_ensemble_data"] = True
        else:
            entry["has_ensemble_data"] = False
            entry["ensemble_dg"] = None

        # Interaction entropy
        if ie_result:
            entry["ie_minus_tds"] = ie_result.get("minus_t_delta_s", None)
            entry["ie_dg"] = ie_result.get("delta_g_ie", None)
        else:
            entry["ie_minus_tds"] = None
            entry["ie_dg"] = None

        # FEP data
        if fep_result:
            entry["fep_method"] = fep_result.get("method", "")
            entry["fep_dg_bind"] = fep_result.get("delta_g_bind", 0.0)
            entry["fep_dg_error"] = fep_result.get("delta_g_error", 0.0)
            entry["fep_corrected_dg"] = fep_result.get("corrected_delta_g", 0.0)
            entry["fep_convergence_passed"] = fep_result.get("convergence_passed", False)
            entry["fep_hysteresis"] = fep_result.get("hysteresis_kcal", 0.0)
            entry["fep_overlap"] = fep_result.get("overlap_minimum", 0.0)
            entry["fep_protein_rmsd"] = fep_result.get("max_protein_rmsd", 0.0)
            entry["fep_passed_qc"] = fep_result.get("passed_qc", False)
            entry["fep_classification"] = fep_result.get("classification", "")
            entry["spike_pharmacophore_match"] = fep_result.get(
                "spike_pharmacophore_match", ""
            )
            entry["fep_dg"] = f"{entry['fep_dg_bind']:.2f} +/- {entry['fep_dg_error']:.2f}"
            entry["has_fep_data"] = True
            entry["classification"] = entry["fep_classification"]
        else:
            entry["has_fep_data"] = False
            entry["fep_dg"] = None
            entry["classification"] = None
            entry["spike_pharmacophore_match"] = None

        self._candidates.append(entry)

    def set_pocket_data(
        self,
        solvent_result: Optional[Dict[str, Any]] = None,
        water_map: Optional[Dict[str, Any]] = None,
        pocket_dynamics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inject pocket-level analysis results."""
        pd: Dict[str, Any] = {"has_pocket_data": False}
        if solvent_result:
            pd["has_pocket_data"] = True
            pd["pocket_stability"] = (
                "STABLE" if solvent_result.get("pocket_stable", False) else "COLLAPSED"
            )
            pd["pocket_rmsd_mean"] = solvent_result.get("pocket_rmsd_mean", 0.0)
            pd["pocket_rmsd"] = pd["pocket_rmsd_mean"]
            pd["pocket_volume_mean"] = solvent_result.get("pocket_volume_mean", 0.0)
            pd["n_structural_waters"] = solvent_result.get("n_structural_waters", 0)
        if water_map:
            pd["has_pocket_data"] = True
            pd["n_displaceable_waters"] = water_map.get("n_displaceable", 0)
            pd["total_displacement_energy"] = water_map.get(
                "total_displacement_energy", 0.0
            )
            pd["displacement_energy"] = pd["total_displacement_energy"]
        else:
            pd["n_displaceable_waters"] = None
            pd["displacement_energy"] = None
        if pocket_dynamics:
            pd["has_pocket_data"] = True
            pd["p_open"] = pocket_dynamics.get("p_open", None)
            pd["p_open_error"] = pocket_dynamics.get("p_open_error", 0.0)
            pd["druggability_classification"] = pocket_dynamics.get(
                "druggability_classification", ""
            )
        else:
            pd["p_open"] = None
            pd["p_open_error"] = 0.0
            pd["druggability_classification"] = ""
        self._pocket_data = pd

    def set_audit_trail(self, audit_dict: Dict[str, Any]) -> None:
        """Inject audit trail data for the campaign report."""
        self._audit_data = audit_dict
        self._stages = audit_dict.get("stages", [])

    def set_generation_stats(self, n_generated: int, n_filtered: int) -> None:
        """Set molecule generation and filtering counts."""
        self._n_generated = n_generated
        self._n_filtered = n_filtered

    # ── Report generation ─────────────────────────────────────────────

    def generate_hit_report(self, candidate: Dict[str, Any]) -> str:
        """Render a single hit report and return the markdown content."""
        ctx = {
            "target_name": self.target_name,
            "pdb_id": self.pdb_id,
            "timestamp": _utcnow(),
            **self._pocket_data,
            **candidate,
        }
        if self._use_jinja:
            return self._hit_template.render(**ctx)
        return _FallbackRenderer.render("hit_report.md.j2", ctx)

    def generate_campaign_report(self) -> str:
        """Render the full campaign summary and return the markdown content."""
        # Classification counts
        classifications = [
            c.get("classification", "PENDING") or "PENDING" for c in self._candidates
        ]
        classification_counts = dict(Counter(classifications))

        # Novel hits
        novel_hits = []
        for c in self._candidates:
            if c.get("classification") == "NOVEL_HIT":
                novel_hits.append({
                    "compound_id": c["compound_id"],
                    "smiles": c["smiles"],
                    "fep_dg_bind": c.get("fep_dg_bind", 0.0),
                    "fep_dg_error": c.get("fep_dg_error", 0.0),
                    "spike_pharmacophore_match": c.get("spike_pharmacophore_match", ""),
                    "tanimoto": c["tanimoto_to_nearest_known"],
                    "nearest_cid": c["nearest_known_cid"],
                })

        # Stage records (simplified)
        stage_records = []
        total_dur = 0.0
        for s in self._stages:
            dur = s.get("duration_seconds", 0.0)
            total_dur += dur
            stage_records.append({
                "stage_name": s.get("stage_name", ""),
                "status": s.get("status", ""),
                "duration_seconds": dur,
            })

        ctx = {
            "project_name": self.project_name,
            "target_name": self.target_name,
            "pdb_id": self.pdb_id,
            "timestamp": _utcnow(),
            "stages": stage_records,
            "total_duration_seconds": total_dur,
            "audit_clean": self._audit_data.get("anti_leakage_clean", True),
            "audit_violations": self._audit_data.get("anti_leakage_violations", []),
            "input_pdb_hash": self._audit_data.get("input_pdb_hash", ""),
            "n_generated": self._n_generated,
            "n_filtered": self._n_filtered,
            "top_n": len(self._candidates),
            "candidates": self._candidates,
            "novel_hits": novel_hits,
            "classification_counts": classification_counts,
            "residue_mappings": [],
            **self._pocket_data,
        }
        if self._use_jinja:
            return self._campaign_template.render(**ctx)
        return _FallbackRenderer.render("campaign_report.md.j2", ctx)

    def generate_all(self) -> Dict[str, str]:
        """Generate all reports and write them to output_dir.

        Returns:
            Dict mapping report name to file path.
        """
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}

        # Per-compound hit reports
        for candidate in self._candidates:
            cid = candidate.get("compound_id", "unknown")
            safe_cid = cid.replace("/", "_").replace(" ", "_")
            content = self.generate_hit_report(candidate)
            out_path = reports_dir / f"hit_{safe_cid}.md"
            out_path.write_text(content, encoding="utf-8")
            paths[f"hit_{safe_cid}"] = str(out_path)

        # Campaign summary
        content = self.generate_campaign_report()
        out_path = reports_dir / "campaign_summary.md"
        out_path.write_text(content, encoding="utf-8")
        paths["campaign_summary"] = str(out_path)

        # Also save structured JSON for programmatic consumption
        summary_json = {
            "target_name": self.target_name,
            "pdb_id": self.pdb_id,
            "n_candidates": len(self._candidates),
            "candidates": self._candidates,
            "pocket_data": self._pocket_data,
            "audit_clean": self._audit_data.get("anti_leakage_clean", True),
        }
        json_path = reports_dir / "campaign_data.json"
        json_path.write_text(
            json.dumps(summary_json, indent=2, default=str), encoding="utf-8"
        )
        paths["campaign_data_json"] = str(json_path)

        logger.info("Generated %d reports in %s", len(paths), reports_dir)
        return paths
