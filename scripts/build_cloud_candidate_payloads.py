#!/usr/bin/env python3
"""Build Cloudflare D1 and Vectorize payloads for top GFlowNet candidates."""

from __future__ import annotations

import hashlib
import json
import math
import argparse
from pathlib import Path
from typing import Any

import polars as pl


REPO = Path("/home/diddy/Desktop/Prism4D-bio")
TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
TOP100 = TRACK_A / "gflownet_top_100_candidates.parquet"
OUT_DIR = TRACK_A / "cloud_payloads"
SQL_PATH = OUT_DIR / "gflownet_candidates_top100.sql"
VECTOR_PATH = OUT_DIR / "gflownet_candidates_top100_vectors.ndjson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=TOP100)
    parser.add_argument("--sql-output", type=Path, default=SQL_PATH)
    parser.add_argument("--vector-output", type=Path, default=VECTOR_PATH)
    parser.add_argument("--training-epoch", type=int, default=500)
    return parser.parse_args()


def sql_quote(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "NULL"
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def candidate_id(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def deterministic_vector(smiles: str, row: dict[str, Any], dimensions: int = 768) -> list[float]:
    """Create a deterministic feature-hash vector for Vectorize indexing.

    This is not a learned Fiber-Bundle embedding. The metadata explicitly marks
    the method so downstream consumers do not confuse it for model latent space.
    """

    seed = hashlib.sha512(smiles.encode("utf-8")).digest()
    base_features = [
        float(row.get("reward") or 0.0),
        float(row.get("pi_complement") or 0.0),
        float(row.get("pi_clash_pocket") or row.get("adjusted_pi_clash") or 0.0),
        float(row.get("pi_clash_lock") or 0.0),
        float(row.get("cryptic_bonus") or 0.0),
        float(row.get("trajectory_entropy") or 0.0),
        float(row.get("policy_logprob") or 0.0),
    ]
    vector: list[float] = []
    for idx in range(dimensions):
        byte = seed[idx % len(seed)]
        signed = (float(byte) / 127.5) - 1.0
        feature = base_features[idx % len(base_features)]
        vector.append((0.85 * signed) + (0.15 * math.tanh(feature)))
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 8) for value in vector]


def numeric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    parsed = float(value)
    return parsed if math.isfinite(parsed) else default


def main() -> int:
    args = parse_args()
    args.sql_output.parent.mkdir(parents=True, exist_ok=True)
    args.vector_output.parent.mkdir(parents=True, exist_ok=True)
    df = pl.scan_parquet(args.candidates).collect()
    rows = df.to_dicts()
    schema_sql = """
DROP TABLE IF EXISTS gflownet_candidates;
CREATE TABLE gflownet_candidates (
    id TEXT PRIMARY KEY,
    smiles TEXT NOT NULL UNIQUE,
    oracle_reward REAL NOT NULL,
    pi_complement REAL,
    pi_clash_pocket REAL,
    pi_clash_lock REAL,
    cryptic_bonus REAL,
    reaction_1 TEXT,
    reaction_2 TEXT,
    synthon_a_id TEXT,
    synthon_b_id TEXT,
    training_epoch INTEGER,
    sampling_temperature REAL,
    bias_projection_score REAL,
    epistemic_confidence TEXT,
    pgx_tier1_worst_case REAL,
    pgx_tier1_mean REAL,
    pgx_population_coverage_pct REAL,
    pgx_ancestry_parity REAL,
    embedding_method TEXT,
    audit_status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_gflownet_candidates_reward ON gflownet_candidates(oracle_reward DESC);
CREATE INDEX IF NOT EXISTS idx_gflownet_candidates_audit ON gflownet_candidates(audit_status);
"""
    statements = [schema_sql.strip()]
    with args.vector_output.open("w", encoding="utf-8") as vector_handle:
        for row in rows:
            smiles = str(row["canonical_smiles"])
            cid = candidate_id(smiles)
            values = {
                "id": cid,
                "smiles": smiles,
                "oracle_reward": float(row["reward"]),
                "pi_complement": float(row["pi_complement"]),
                "pi_clash_pocket": float(row.get("pi_clash_pocket") or row["adjusted_pi_clash"]),
                "pi_clash_lock": float(row.get("pi_clash_lock") or 0.0),
                "cryptic_bonus": float(row["cryptic_bonus"]),
                "reaction_1": "policy_sampled_anchor_lookup",
                "reaction_2": "",
                "synthon_a_id": str(row.get("anchor_id") or ""),
                "synthon_b_id": "",
                "training_epoch": int(args.training_epoch),
                "sampling_temperature": numeric(row, "sampling_temperature"),
                "bias_projection_score": numeric(row, "bias_projection_score"),
                "epistemic_confidence": str(row.get("epistemic_confidence") or "L1"),
                "pgx_tier1_worst_case": numeric(row, "pgx_tier1_worst_case"),
                "pgx_tier1_mean": numeric(row, "pgx_tier1_mean"),
                "pgx_population_coverage_pct": numeric(row, "pgx_population_coverage_pct"),
                "pgx_ancestry_parity": numeric(row, "pgx_ancestry_parity"),
                "embedding_method": "deterministic_feature_hash_projection_v0",
                "audit_status": "pending",
            }
            statements.append(
                "INSERT OR REPLACE INTO gflownet_candidates "
                "(id, smiles, oracle_reward, pi_complement, pi_clash_pocket, pi_clash_lock, cryptic_bonus, "
                "reaction_1, reaction_2, synthon_a_id, synthon_b_id, training_epoch, "
                "sampling_temperature, bias_projection_score, epistemic_confidence, "
                "pgx_tier1_worst_case, pgx_tier1_mean, pgx_population_coverage_pct, pgx_ancestry_parity, "
                "embedding_method, audit_status) VALUES "
                "(" + ", ".join(sql_quote(values[key]) for key in values) + ");"
            )
            vector_handle.write(
                json.dumps(
                    {
                        "id": cid,
                        "values": deterministic_vector(smiles, row),
                        "metadata": {
                            "smiles": smiles,
                            "reward": float(row["reward"]),
                            "bias_projection_score": numeric(row, "bias_projection_score"),
                            "pgx_tier1_worst_case": numeric(row, "pgx_tier1_worst_case"),
                            "pgx_population_coverage_pct": numeric(row, "pgx_population_coverage_pct"),
                            "reaction_route": "policy_sampled_anchor_lookup",
                            "embedding_method": "deterministic_feature_hash_projection_v0",
                        },
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    args.sql_output.write_text("\n".join(statements) + "\n", encoding="utf-8")
    print(
        "cloud_payloads_built "
        f"candidates={len(rows)} sql={args.sql_output} vector_ndjson={args.vector_output} "
        "embedding_method=deterministic_feature_hash_projection_v0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
