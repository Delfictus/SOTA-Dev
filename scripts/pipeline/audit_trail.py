"""Anti-leakage audit trail for PRISM-4D pipeline.

Records every pipeline stage with:
- Timestamps (UTC ISO-8601)
- Input file hashes (SHA-256)
- Stage outcomes and metadata
- External database queries (post-detection only)
- Classification justification

The audit trail proves that PRISM ran *blind* (no external data used during
detection) and that external databases were consulted only AFTER spike
detection was complete.

Usage::

    trail = AuditTrail(output_dir="/tmp/campaign")
    trail.start_pipeline(pdb_path="input.pdb")
    trail.log_stage("target_classification", inputs=["input.pdb"],
                    result={"classification": "soluble"})
    ...
    trail.finalize()
    trail.save()
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Stages that run BEFORE external data is allowed.
# Any external DB query during these stages is an anti-leakage violation.
BLIND_STAGES = frozenset({
    "target_classification",
    "protein_fixing",
    "prism_detection",
    "pocket_refinement",
    "water_map",
    "pocket_popen",
    "pharmacophore",
})

# Stages where external DB queries are permitted (post-detection).
EXTERNAL_ALLOWED_STAGES = frozenset({
    "membrane_building",       # OPM database lookup is structural, not ligand
    "generation",
    "tautomer_enumeration",
    "filtering",               # PubChem novelty check
    "docking",
    "ensemble_scoring",
    "fep",
    "reporting",
})


def _sha256(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return f"UNREADABLE:{path}"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageRecord:
    """Record of a single pipeline stage execution."""

    stage_name: str
    status: str                            # "started" | "completed" | "failed" | "skipped"
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    input_hashes: Dict[str, str] = field(default_factory=dict)
    output_hashes: Dict[str, str] = field(default_factory=dict)
    result_summary: Dict[str, Any] = field(default_factory=dict)
    external_queries: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    conda_env: str = ""
    command: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditTrail:
    """Full pipeline audit trail."""

    output_dir: str
    pipeline_id: str = field(default_factory=lambda: _utcnow().replace(":", "-"))
    pipeline_start: str = ""
    pipeline_end: str = ""
    input_pdb_hash: str = ""
    input_pdb_path: str = ""
    config_hash: str = ""
    stages: List[StageRecord] = field(default_factory=list)
    anti_leakage_violations: List[str] = field(default_factory=list)
    _current_stage: Optional[StageRecord] = field(default=None, repr=False)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start_pipeline(self, pdb_path: str, config_path: str = "") -> None:
        """Mark pipeline start and hash the primary input."""
        self.pipeline_start = _utcnow()
        self.input_pdb_path = str(pdb_path)
        self.input_pdb_hash = _sha256(pdb_path)
        if config_path:
            self.config_hash = _sha256(config_path)
        logger.info("Audit trail started: pipeline_id=%s", self.pipeline_id)

    def finalize(self) -> None:
        """Mark pipeline end and run anti-leakage verification."""
        self.pipeline_end = _utcnow()
        self._verify_anti_leakage()
        logger.info(
            "Audit trail finalized: %d stages, %d violations",
            len(self.stages),
            len(self.anti_leakage_violations),
        )

    # ── Stage logging ─────────────────────────────────────────────────

    def begin_stage(
        self,
        stage_name: str,
        inputs: Optional[List[str]] = None,
        conda_env: str = "",
        command: str = "",
    ) -> None:
        """Begin recording a pipeline stage."""
        rec = StageRecord(
            stage_name=stage_name,
            status="started",
            start_time=_utcnow(),
            conda_env=conda_env,
            command=command,
        )
        if inputs:
            for p in inputs:
                rec.input_hashes[str(p)] = _sha256(p)
        self._current_stage = rec
        logger.info("Stage started: %s", stage_name)

    def end_stage(
        self,
        status: str = "completed",
        result: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
    ) -> None:
        """Finish recording the current pipeline stage."""
        rec = self._current_stage
        if rec is None:
            logger.warning("end_stage called with no active stage")
            return
        rec.status = status
        rec.end_time = _utcnow()
        # Compute duration
        try:
            t0 = datetime.fromisoformat(rec.start_time)
            t1 = datetime.fromisoformat(rec.end_time)
            rec.duration_seconds = round((t1 - t0).total_seconds(), 3)
        except (ValueError, TypeError):
            rec.duration_seconds = 0.0
        if result:
            rec.result_summary = result
        if outputs:
            for p in outputs:
                rec.output_hashes[str(p)] = _sha256(p)
        if errors:
            rec.errors = errors
        self.stages.append(rec)
        self._current_stage = None
        logger.info("Stage %s: %s (%.1fs)", status, rec.stage_name, rec.duration_seconds)

    def skip_stage(self, stage_name: str, reason: str = "") -> None:
        """Record a skipped stage."""
        rec = StageRecord(
            stage_name=stage_name,
            status="skipped",
            start_time=_utcnow(),
            end_time=_utcnow(),
            result_summary={"skip_reason": reason},
        )
        self.stages.append(rec)
        logger.info("Stage skipped: %s (%s)", stage_name, reason)

    def log_stage(
        self,
        stage_name: str,
        inputs: Optional[List[str]] = None,
        result: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[str]] = None,
        conda_env: str = "",
        command: str = "",
    ) -> None:
        """Convenience: log a stage that completed instantly (e.g. in dry-run)."""
        self.begin_stage(stage_name, inputs=inputs, conda_env=conda_env, command=command)
        self.end_stage(status="completed", result=result, outputs=outputs)

    def log_external_query(self, database: str, query: str) -> None:
        """Record an external database query for anti-leakage verification."""
        rec = self._current_stage
        entry = f"{database}: {query}"
        if rec is not None:
            rec.external_queries.append(entry)
            if rec.stage_name in BLIND_STAGES:
                violation = (
                    f"ANTI-LEAKAGE VIOLATION: External query '{entry}' "
                    f"during blind stage '{rec.stage_name}'"
                )
                self.anti_leakage_violations.append(violation)
                logger.error(violation)
        else:
            logger.warning("External query logged with no active stage: %s", entry)

    # ── Anti-leakage verification ─────────────────────────────────────

    def _verify_anti_leakage(self) -> None:
        """Scan all stages for external queries during blind phases."""
        for rec in self.stages:
            if rec.stage_name in BLIND_STAGES and rec.external_queries:
                for q in rec.external_queries:
                    violation = (
                        f"ANTI-LEAKAGE VIOLATION: External query '{q}' "
                        f"during blind stage '{rec.stage_name}'"
                    )
                    if violation not in self.anti_leakage_violations:
                        self.anti_leakage_violations.append(violation)

    @property
    def is_clean(self) -> bool:
        """True if no anti-leakage violations were detected."""
        return len(self.anti_leakage_violations) == 0

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_start": self.pipeline_start,
            "pipeline_end": self.pipeline_end,
            "input_pdb_path": self.input_pdb_path,
            "input_pdb_hash": self.input_pdb_hash,
            "config_hash": self.config_hash,
            "anti_leakage_clean": self.is_clean,
            "anti_leakage_violations": self.anti_leakage_violations,
            "stages": [s.to_dict() for s in self.stages],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Optional[str] = None) -> str:
        """Write audit trail JSON to disk. Returns the path written."""
        if path is None:
            out = Path(self.output_dir) / "audit_trail.json"
        else:
            out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        logger.info("Audit trail saved: %s", out)
        return str(out)

    @classmethod
    def from_json_file(cls, path: str) -> AuditTrail:
        """Load an existing audit trail from JSON."""
        with open(path) as f:
            d = json.load(f)
        trail = cls(output_dir=str(Path(path).parent))
        trail.pipeline_id = d.get("pipeline_id", "")
        trail.pipeline_start = d.get("pipeline_start", "")
        trail.pipeline_end = d.get("pipeline_end", "")
        trail.input_pdb_path = d.get("input_pdb_path", "")
        trail.input_pdb_hash = d.get("input_pdb_hash", "")
        trail.config_hash = d.get("config_hash", "")
        trail.anti_leakage_violations = d.get("anti_leakage_violations", [])
        trail.stages = [StageRecord(**s) for s in d.get("stages", [])]
        return trail
