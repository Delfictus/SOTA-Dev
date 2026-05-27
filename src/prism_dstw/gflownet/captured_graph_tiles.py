"""Captured graph tile definitions and registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from prism_dstw.gflownet.bsr_operator_state import BSROperatorState
from prism_dstw.gflownet.tile_operator_delta import TileOperatorDelta


ALLOWED_TILE_TYPES = {
    "holo_tile_fusion",
    "lock_interface_destabilization",
    "hydration_channel_preservation",
    "nma_hinge_preservation",
    "arrestin_boundary_avoidance",
    "camp_basin_stabilization",
    "quiet_lock_disruption",
}


@dataclass(frozen=True)
class CapturedGraphTile:
    tile_id: str
    tile_type: str
    topology_region: str
    perturbation_family: str
    affected_voxel_ids: tuple[int, ...]
    affected_state_ids: tuple[int, ...]
    affected_bsr_blocks: tuple[int, ...]
    delta_values: tuple[tuple[tuple[float, ...], ...], ...]
    restricted_operator_target: str
    capture_shape_bucket: str
    cuda_graph_id: str | None
    tile_delta_hash: str
    provenance_hash: str
    topology_delta: str
    basin_delta: str
    restricted_operator_hash: str
    c6_operator_hash: str
    captured_graph_tile_hash: str

    def to_delta(self, *, state: BSROperatorState) -> TileOperatorDelta:
        return TileOperatorDelta.from_lists(
            tile_id=self.tile_id,
            affected_bsr_blocks=list(self.affected_bsr_blocks),
            delta_values=[
                [[float(value) for value in row] for row in block]
                for block in self.delta_values
            ],
            topology_delta=self.topology_delta,
            basin_delta=self.basin_delta,
            restricted_operator_target=self.restricted_operator_target,
            device=state.device,
            dtype=state.dtype,
        )

    def with_cuda_graph_id(self, cuda_graph_id: str) -> "CapturedGraphTile":
        data = asdict(self)
        data["cuda_graph_id"] = cuda_graph_id
        return CapturedGraphTile(**data)

    def without_capture(self) -> "CapturedGraphTile":
        data = asdict(self)
        data["cuda_graph_id"] = None
        return CapturedGraphTile(**data)


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_tile(
    *,
    tile_id: str,
    tile_type: str,
    topology_region: str,
    perturbation_family: str,
    affected_voxel_ids: list[int],
    affected_state_ids: list[int],
    affected_bsr_blocks: list[int],
    delta_values: list[list[list[float]]],
    restricted_operator_target: str,
    capture_shape_bucket: str,
    topology_delta: str,
    basin_delta: str,
    restricted_operator_hash: str,
    c6_operator_hash: str,
) -> CapturedGraphTile:
    if tile_type not in ALLOWED_TILE_TYPES:
        raise ValueError(f"unsupported tile_type={tile_type}")
    delta_payload = {
        "tile_id": tile_id,
        "affected_bsr_blocks": affected_bsr_blocks,
        "delta_values": delta_values,
        "topology_delta": topology_delta,
        "basin_delta": basin_delta,
    }
    tile_delta_hash = _canonical_hash(delta_payload)
    provenance_hash = _canonical_hash(
        {
            "tile_id": tile_id,
            "tile_type": tile_type,
            "topology_region": topology_region,
            "perturbation_family": perturbation_family,
            "restricted_operator_target": restricted_operator_target,
        }
    )
    captured_graph_tile_hash = _canonical_hash(
        {
            "tile_delta_hash": tile_delta_hash,
            "provenance_hash": provenance_hash,
            "restricted_operator_hash": restricted_operator_hash,
            "c6_operator_hash": c6_operator_hash,
        }
    )
    return CapturedGraphTile(
        tile_id=tile_id,
        tile_type=tile_type,
        topology_region=topology_region,
        perturbation_family=perturbation_family,
        affected_voxel_ids=tuple(int(value) for value in affected_voxel_ids),
        affected_state_ids=tuple(int(value) for value in affected_state_ids),
        affected_bsr_blocks=tuple(int(value) for value in affected_bsr_blocks),
        delta_values=tuple(tuple(tuple(float(value) for value in row) for row in block) for block in delta_values),
        restricted_operator_target=restricted_operator_target,
        capture_shape_bucket=capture_shape_bucket,
        cuda_graph_id=None,
        tile_delta_hash=tile_delta_hash,
        provenance_hash=provenance_hash,
        topology_delta=topology_delta,
        basin_delta=basin_delta,
        restricted_operator_hash=restricted_operator_hash,
        c6_operator_hash=c6_operator_hash,
        captured_graph_tile_hash=captured_graph_tile_hash,
    )


def _expected_hashes(row: dict[str, Any]) -> tuple[str, str, str]:
    delta_payload = {
        "tile_id": str(row["tile_id"]),
        "affected_bsr_blocks": [int(value) for value in row["affected_bsr_blocks"]],
        "delta_values": row["delta_values"],
        "topology_delta": str(row["topology_delta"]),
        "basin_delta": str(row["basin_delta"]),
    }
    tile_delta_hash = _canonical_hash(delta_payload)
    provenance_hash = _canonical_hash(
        {
            "tile_id": str(row["tile_id"]),
            "tile_type": str(row["tile_type"]),
            "topology_region": str(row["topology_region"]),
            "perturbation_family": str(row["perturbation_family"]),
            "restricted_operator_target": str(row["restricted_operator_target"]),
        }
    )
    captured_graph_tile_hash = _canonical_hash(
        {
            "tile_delta_hash": tile_delta_hash,
            "provenance_hash": provenance_hash,
            "restricted_operator_hash": str(row["restricted_operator_hash"]),
            "c6_operator_hash": str(row["c6_operator_hash"]),
        }
    )
    return tile_delta_hash, provenance_hash, captured_graph_tile_hash


@dataclass
class CapturedGraphTileRegistry:
    tiles: dict[str, CapturedGraphTile]
    path: Path | None = None

    @classmethod
    def from_json(cls, path: Path) -> "CapturedGraphTileRegistry":
        payload = json.loads(path.read_text())
        schema_version = payload.get("schema_version")
        if schema_version != "prism.log_subtb.captured_tile_registry.v1":
            raise ValueError(f"{path}: unsupported or missing schema_version {schema_version!r}")
        tiles: dict[str, CapturedGraphTile] = {}
        for row in payload.get("tiles", []):
            expected_delta, expected_provenance, expected_captured = _expected_hashes(row)
            if str(row["tile_delta_hash"]) != expected_delta:
                raise ValueError(f"{path}: tile_delta_hash mismatch for {row.get('tile_id')}")
            if str(row["provenance_hash"]) != expected_provenance:
                raise ValueError(f"{path}: provenance_hash mismatch for {row.get('tile_id')}")
            if str(row["captured_graph_tile_hash"]) != expected_captured:
                raise ValueError(f"{path}: captured_graph_tile_hash mismatch for {row.get('tile_id')}")
            tile = CapturedGraphTile(
                tile_id=str(row["tile_id"]),
                tile_type=str(row["tile_type"]),
                topology_region=str(row["topology_region"]),
                perturbation_family=str(row["perturbation_family"]),
                affected_voxel_ids=tuple(int(value) for value in row["affected_voxel_ids"]),
                affected_state_ids=tuple(int(value) for value in row["affected_state_ids"]),
                affected_bsr_blocks=tuple(int(value) for value in row["affected_bsr_blocks"]),
                delta_values=tuple(
                    tuple(tuple(float(value) for value in inner) for inner in block)
                    for block in row["delta_values"]
                ),
                restricted_operator_target=str(row["restricted_operator_target"]),
                capture_shape_bucket=str(row["capture_shape_bucket"]),
                cuda_graph_id=None,
                tile_delta_hash=str(row["tile_delta_hash"]),
                provenance_hash=str(row["provenance_hash"]),
                topology_delta=str(row["topology_delta"]),
                basin_delta=str(row["basin_delta"]),
                restricted_operator_hash=str(row["restricted_operator_hash"]),
                c6_operator_hash=str(row["c6_operator_hash"]),
                captured_graph_tile_hash=str(row["captured_graph_tile_hash"]),
            )
            if tile.tile_type not in ALLOWED_TILE_TYPES:
                raise ValueError(f"{path}: unsupported tile type {tile.tile_type}")
            if tile.tile_id in tiles:
                raise ValueError(f"{path}: duplicate tile_id {tile.tile_id}")
            tiles[tile.tile_id] = tile
        return cls(tiles=tiles, path=path)

    @classmethod
    def demo(cls, *, tile_count: int = 7) -> "CapturedGraphTileRegistry":
        tile_types = sorted(ALLOWED_TILE_TYPES)
        tiles: dict[str, CapturedGraphTile] = {}
        for index in range(tile_count):
            tile_type = tile_types[index % len(tile_types)]
            source_block = index
            target_block = index + 1
            tile = build_tile(
                tile_id=f"captured_tile_{index:03d}",
                tile_type=tile_type,
                topology_region=f"REGION_{index % 3}",
                perturbation_family=f"FAMILY_{index % 4}",
                affected_voxel_ids=[1000 + index],
                affected_state_ids=[index, index + 1],
                affected_bsr_blocks=[source_block, target_block],
                delta_values=[[[0.015]], [[-0.005]]],
                restricted_operator_target="W_without_arr(Pi)",
                capture_shape_bucket="rows2_blocks2_block1_float64",
                topology_delta=f"topology_delta_{index}",
                basin_delta=f"basin_delta_{index}",
                restricted_operator_hash=f"restricted_seed_{index}",
                c6_operator_hash=f"c6_seed_{index}",
            )
            tiles[tile.tile_id] = tile
        return cls(tiles=tiles)

    def get(self, tile_id: str) -> CapturedGraphTile:
        try:
            return self.tiles[tile_id]
        except KeyError as exc:
            raise KeyError(f"tile_id not present in captured registry: {tile_id}") from exc

    def validate_for_state(self, state: BSROperatorState) -> list[str]:
        uncaptured: list[str] = []
        for tile in self.tiles.values():
            if tile.cuda_graph_id is None:
                uncaptured.append(tile.tile_id)
            tile.to_delta(state=state).validate_for_state(state)
        return uncaptured

    def with_captures(self, capture_ids: dict[str, str]) -> "CapturedGraphTileRegistry":
        return CapturedGraphTileRegistry(
            tiles={
                tile_id: tile.with_cuda_graph_id(capture_ids[tile_id]) if tile_id in capture_ids else tile
                for tile_id, tile in self.tiles.items()
            },
            path=self.path,
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "prism.log_subtb.captured_tile_registry.v1",
            "tiles": [asdict(tile) for tile in self.tiles.values()],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def manifest(self, *, capture_ids: dict[str, str], state: BSROperatorState, warmup_iterations: int) -> dict[str, Any]:
        captured = len(capture_ids)
        return {
            "schema_version": "prism.log_subtb.tile_capture_manifest.v1",
            "tile_count": len(self.tiles),
            "captured_tile_count": captured,
            "uncaptured_tile_count": len(self.tiles) - captured,
            "capture_bucket_count": len({tile.capture_shape_bucket for tile in self.tiles.values()}),
            "cuda_graph_count": captured,
            "blocksize": list(state.blocksize),
            "dtype": str(state.dtype).replace("torch.", ""),
            "device": str(state.device),
            "warmup_iterations": warmup_iterations,
            "capture_success": captured == len(self.tiles) and captured > 0,
        }
