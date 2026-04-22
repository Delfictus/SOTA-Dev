"""SiteSpikeView — D5 vectorized aggregate layer over the validated triad.

Version: site_spike_view_v1

Triad (all three required):
  - <stem>.topology.spike_events.arrow  (canonical columnar)
  - <stem>.run_metadata.json            (enum tables + master protocol)
  - <stem>.binding_sites.json           (site centroids)

Contract invariants:
  - Spatial membership: site_radius_sq = (lining_cutoff + 2.0) ** 2
  - Phase label: 5-state closure on timestep via reference_protocol_for_json_phase_label
  - Enum decode: aromatic_type + spike_source via run_metadata tables
  - All aggregation methods are vectorized (numpy / polars); no per-spike
    Python dict materialization in the hot path.

This module intentionally does NOT import polars at top-level (kept lazy for
voxel_aggregate) so light-weight consumers pay zero polars startup cost.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SCHEMA_VERSION = "site_spike_view_v1"


_REQUIRED_ARROW_COLUMNS = (
    "x", "y", "z", "intensity", "aromatic_type", "wavelength_nm",
    "spike_source", "aromatic_residue_id", "water_density",
    "vibrational_energy", "n_nearby_excited", "timestep",
    "frame_index", "stream_id",
)

_PHASE_ORDER = ("cold_hold", "heating", "warm_hold", "cooling", "cold_return")


class SiteSpikeViewError(Exception):
    """Raised when a triad is present but malformed or incomplete.

    Absence of triad files is a SOFT fail — from_target_dir returns None in
    that case so callers can fall back to legacy JSON readers.
    """

    def __init__(self, reason: str, missing_files: Optional[List[str]] = None,
                 target_dir: Optional[Path] = None):
        super().__init__(reason)
        self.reason = reason
        self.missing_files = list(missing_files or [])
        self.target_dir = target_dir


class SiteSpikeView:
    """Lossless vectorized access over the canonical triad."""

    def __init__(self, table, meta: dict, bs: dict,
                 arrow_path: Path, meta_path: Path, bs_path: Path,
                 target_dir: Optional[Path] = None, stem: Optional[str] = None):
        self._table = table
        self._meta = meta
        self._bs = bs
        self._arrow_path = arrow_path
        self._meta_path = meta_path
        self._bs_path = bs_path
        self._target_dir = target_dir
        self._stem = stem

        proto = meta["reference_protocol_for_json_phase_label"]
        self._p1 = int(proto["cold_hold_steps"])
        self._p2 = self._p1 + int(proto["ramp_steps"])
        self._p3 = self._p2 + int(proto["warm_hold_steps"])
        self._p4 = self._p3 + int(proto.get("ramp_down_steps", 0))

        self._arom_enum: Dict[int, str] = {int(k): str(v) for k, v in meta["aromatic_type_enum"].items()}
        self._arom_default = str(meta.get("aromatic_type_default", "UNK"))
        self._src_enum: Dict[int, str] = {int(k): str(v) for k, v in meta["spike_source_enum"].items()}
        self._src_default = str(meta.get("spike_source_default", "LIF"))

        self._lining_cutoff = float(meta.get("lining_cutoff", 8.0))
        self._site_radius = self._lining_cutoff + 2.0
        self._site_radius_sq = self._site_radius * self._site_radius

        sites = [s for s in (bs.get("sites") or [])
                 if isinstance(s, dict) and isinstance(s.get("centroid"), list)
                 and len(s.get("centroid")) == 3]
        self._site_centroids: Dict[int, Tuple[float, float, float]] = {
            int(s["id"]): (float(s["centroid"][0]), float(s["centroid"][1]), float(s["centroid"][2]))
            for s in sites if "id" in s
        }
        self._site_ids_sorted = sorted(self._site_centroids.keys())

        # Eager spatial columns for mask computation
        self._x_full = self._table.column("x").to_numpy(zero_copy_only=False)
        self._y_full = self._table.column("y").to_numpy(zero_copy_only=False)
        self._z_full = self._table.column("z").to_numpy(zero_copy_only=False)

        self._slice_cache: Dict[int, "SiteSlice"] = {}

        # D5_v2: view-level full-column cache + LAZY per-site int-index.
        # - Full columns are materialized once and shared across all slices
        #   (eliminates the v1 per-slice re-decode of 66M-row columns).
        # - Site idx is computed on first .site(sid) call and cached.
        #   Single-site consumers (pharmacophore) pay O(N_full) once;
        #   all-site consumers (response_selectivity) pay O(N_full × N_sites)
        #   but amortize via cached full columns.
        self._full_col_cache: Dict[str, np.ndarray] = {}
        self._site_idx: Dict[int, np.ndarray] = {}

    def _full_col(self, name: str) -> np.ndarray:
        """Return the full topology column as a numpy array, cached at view level."""
        if name not in self._full_col_cache:
            self._full_col_cache[name] = self._table.column(name).to_numpy(zero_copy_only=False)
        return self._full_col_cache[name]

    def _site_idx_for(self, sid: int) -> np.ndarray:
        if sid in self._site_idx:
            return self._site_idx[sid]
        cx, cy, cz = self._site_centroids[sid]
        d2 = (self._x_full - cx) ** 2 + (self._y_full - cy) ** 2 + (self._z_full - cz) ** 2
        idx = np.flatnonzero(d2 <= self._site_radius_sq)
        self._site_idx[sid] = idx
        return idx

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def version(cls) -> str:
        return SCHEMA_VERSION

    @classmethod
    def schema_contract(cls) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "required_arrow_columns": list(_REQUIRED_ARROW_COLUMNS),
            "enum_sources": {
                "aromatic_type": "run_metadata.aromatic_type_enum",
                "spike_source": "run_metadata.spike_source_enum",
            },
            "spatial_membership_rule":
                "site_radius_sq = (lining_cutoff + 2.0)**2; "
                "mask = d2 <= site_radius_sq (legacy JSON semantics, NOT Arrow site_id)",
            "phase_label_rule":
                "5-state closure on timestep via reference_protocol_for_json_phase_label",
            "tolerance_rules": {
                "integer_counts": "exact",
                "float_sums_means": "<=1e-6 relative",
                "sharpness_energy_density": "<=1e-5 relative",
            },
            "gate_a_validated_targets": ["m1_2akr", "m1_3umi"],
        }

    @classmethod
    def from_target_dir(cls, target_dir, stem: str) -> Optional["SiteSpikeView"]:
        td = Path(target_dir)
        eng = td / "artifacts/5_engine"
        arrow_p = eng / f"{stem}.topology.spike_events.arrow"
        meta_p = eng / f"{stem}.run_metadata.json"
        bs_p = eng / f"{stem}.binding_sites.json"
        missing = [str(p) for p in (arrow_p, meta_p, bs_p) if not p.exists()]
        if missing:
            return None
        try:
            return cls.from_triad(arrow_p, meta_p, bs_p, target_dir=td, stem=stem)
        except SiteSpikeViewError:
            raise

    @classmethod
    def from_triad(cls, arrow_path, run_metadata_path, binding_sites_path,
                   target_dir=None, stem: Optional[str] = None) -> "SiteSpikeView":
        arrow_p = Path(arrow_path)
        meta_p = Path(run_metadata_path)
        bs_p = Path(binding_sites_path)
        missing = [str(p) for p in (arrow_p, meta_p, bs_p) if not p.exists()]
        if missing:
            raise SiteSpikeViewError(
                f"triad files missing: {missing}",
                missing_files=missing,
                target_dir=target_dir,
            )

        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.ipc as ipc
        except Exception as e:
            raise SiteSpikeViewError(f"pyarrow unavailable: {e}")

        try:
            meta = json.loads(meta_p.read_text())
        except Exception as e:
            raise SiteSpikeViewError(f"run_metadata.json malformed: {e}")

        proto = meta.get("reference_protocol_for_json_phase_label")
        if not isinstance(proto, dict) or "error" in proto:
            raise SiteSpikeViewError(
                f"reference_protocol_for_json_phase_label missing/invalid: {proto}"
            )
        for k in ("cold_hold_steps", "ramp_steps", "warm_hold_steps"):
            if not isinstance(proto.get(k), int):
                raise SiteSpikeViewError(f"run_metadata protocol.{k} not int")

        arom_enum = meta.get("aromatic_type_enum")
        src_enum = meta.get("spike_source_enum")
        if not isinstance(arom_enum, dict) or not arom_enum:
            raise SiteSpikeViewError("aromatic_type_enum missing or empty")
        if not isinstance(src_enum, dict) or not src_enum:
            raise SiteSpikeViewError("spike_source_enum missing or empty")
        if not isinstance(meta.get("lining_cutoff"), (int, float)):
            raise SiteSpikeViewError("lining_cutoff missing/invalid")

        try:
            bs = json.loads(bs_p.read_text())
        except Exception as e:
            raise SiteSpikeViewError(f"binding_sites.json malformed: {e}")
        if not isinstance(bs.get("sites"), list):
            raise SiteSpikeViewError("binding_sites.sites missing or not a list")

        try:
            src = __import__("pyarrow").memory_map(str(arrow_p), "rb")
        except Exception as e:
            raise SiteSpikeViewError(f"pyarrow.memory_map failed: {e}")
        try:
            try:
                reader = ipc.open_file(src)
            except Exception:
                # Fall back to stream format
                src.close()
                src = __import__("pyarrow").memory_map(str(arrow_p), "rb")
                reader = ipc.open_stream(src)
            table = reader.read_all()
        except Exception as e:
            raise SiteSpikeViewError(f"Arrow read failed: {e}")

        for col in _REQUIRED_ARROW_COLUMNS:
            if col not in table.column_names:
                raise SiteSpikeViewError(f"Arrow missing required column: {col}")

        return cls(table, meta, bs, arrow_p, meta_p, bs_p,
                   target_dir=target_dir, stem=stem)

    def version(self) -> str:  # type: ignore[override]
        return SCHEMA_VERSION

    def schema_contract(self) -> dict:  # type: ignore[override]
        base = type(self).schema_contract()
        base.update({
            "target_dir": str(self._target_dir) if self._target_dir else None,
            "stem": self._stem,
            "arrow_path": str(self._arrow_path),
            "run_metadata_path": str(self._meta_path),
            "binding_sites_path": str(self._bs_path),
            "n_rows_in_arrow": int(self._table.num_rows),
            "n_sites": len(self._site_centroids),
            "site_ids_discovered": list(self._site_ids_sorted),
            "lining_cutoff": self._lining_cutoff,
            "site_radius": self._site_radius,
        })
        return base

    # ── Discovery ───────────────────────────────────────────────────────

    def available_site_ids(self) -> List[int]:
        return list(self._site_ids_sorted)

    def site_ids(self) -> List[int]:
        return list(self._site_ids_sorted)

    def has_site(self, sid: int) -> bool:
        return int(sid) in self._site_centroids

    def export_site_parquet(self, sid: int, path, compression: str = "zstd"):
        """Write a per-site parquet for `sid` matching the legacy per-site parquet
        schema (Arrow → parquet, no JSON intermediate). Columns: all 14 raw +
        'type' (decoded aromatic_type), 'ccns_phase' (decoded), plus
        spike_source also decoded as 'spike_source_name'. Returns the Path
        written.

        Used by the sync pipeline (prism_spike_watcher) to replace the
        JSON→parquet converter once engine-JSON emission is turned off.

        Non-hot-path: this is an I/O writer, not a query method.
        """
        try:
            import polars as pl
        except Exception as e:
            raise SiteSpikeViewError(f"polars required for export_site_parquet: {e}")
        sl = self.site(sid)
        if sl is None:
            raise SiteSpikeViewError(f"site {sid} not in binding_sites")
        df = sl.to_polars(columns=None)
        # Append decoded string columns under conventional legacy names
        df = df.with_columns([
            pl.Series("type", sl.aromatic_type_decoded()),
            pl.Series("spike_source_name", sl.spike_source_decoded()),
        ])
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out, compression=compression)
        return out

    def site(self, sid: int) -> Optional["SiteSlice"]:
        key = int(sid)
        if key in self._slice_cache:
            return self._slice_cache[key]
        if key not in self._site_centroids:
            return None
        # v2: lazy per-site idx; amortizes across repeated .site(key) calls
        idx = self._site_idx_for(key)
        slice_obj = SiteSlice(self, key, idx)
        self._slice_cache[key] = slice_obj
        return slice_obj


class SiteSlice:
    """Per-site vectorized slice. Hot-path numpy; no per-spike dicts.

    v2: holds a precomputed sorted int-index array (not a boolean mask).
    Column access is an integer gather on the view-level cached full column.
    """

    def __init__(self, view: SiteSpikeView, sid: int, idx: np.ndarray):
        self._view = view
        self._sid = int(sid)
        self._idx = idx  # int64 row indices (sorted ascending)
        self._n_spikes = int(idx.size)
        self._col_cache: Dict[str, np.ndarray] = {}
        self._agg_cache: Dict[str, object] = {}
        self._voxel_cache: Dict[float, dict] = {}

    # ── Metadata ────────────────────────────────────────────────────────

    def site_id(self) -> int:
        return self._sid

    def centroid(self) -> Tuple[float, float, float]:
        return self._view._site_centroids[self._sid]

    def n_spikes(self) -> int:
        return self._n_spikes

    def n_spikes_analyzed(self) -> int:
        return self._n_spikes

    def lining_cutoff(self) -> float:
        return self._view._lining_cutoff

    def site_radius(self) -> float:
        return self._view._site_radius

    def n_frames(self) -> int:
        if self._n_spikes == 0:
            return 0
        if "__n_frames" in self._agg_cache:
            return self._agg_cache["__n_frames"]  # type: ignore[return-value]
        frames = self.frame_index()
        n = int(np.unique(frames).size)
        self._agg_cache["__n_frames"] = n
        return n

    def open_frequency(self) -> float:
        if self._n_spikes == 0:
            return 0.0
        if "__open_frequency" in self._agg_cache:
            return self._agg_cache["__open_frequency"]  # type: ignore[return-value]
        frames = self.frame_index()
        unique = int(np.unique(frames).size)
        max_frame = int(frames.max())
        ofreq = unique / max(max_frame + 1, 1)
        self._agg_cache["__open_frequency"] = ofreq
        return ofreq

    def stream_ids_present(self) -> List[int]:
        if self._n_spikes == 0:
            return []
        if "__streams" in self._agg_cache:
            return list(self._agg_cache["__streams"])  # type: ignore[arg-type]
        s = sorted(int(x) for x in np.unique(self.stream_id()).tolist())
        self._agg_cache["__streams"] = s
        return list(s)

    def phase_labels_present(self) -> List[str]:
        pc = self.phase_counts()
        return [name for name in _PHASE_ORDER if name in pc]

    # ── Raw columns (lazy, cached) ──────────────────────────────────────

    def _col(self, name: str) -> np.ndarray:
        if name not in self._col_cache:
            # v2: full column cached at view level; integer-gather is O(n_site)
            self._col_cache[name] = self._view._full_col(name)[self._idx]
        return self._col_cache[name]

    def x(self) -> np.ndarray: return self._col("x")
    def y(self) -> np.ndarray: return self._col("y")
    def z(self) -> np.ndarray: return self._col("z")
    def intensity(self) -> np.ndarray: return self._col("intensity")
    def wavelength_nm(self) -> np.ndarray: return self._col("wavelength_nm")
    def aromatic_type(self) -> np.ndarray: return self._col("aromatic_type")
    def spike_source(self) -> np.ndarray: return self._col("spike_source")
    def aromatic_residue_id(self) -> np.ndarray: return self._col("aromatic_residue_id")
    def water_density(self) -> np.ndarray: return self._col("water_density")
    def vibrational_energy(self) -> np.ndarray: return self._col("vibrational_energy")
    def n_nearby_excited(self) -> np.ndarray: return self._col("n_nearby_excited")
    def timestep(self) -> np.ndarray: return self._col("timestep")
    def frame_index(self) -> np.ndarray: return self._col("frame_index")
    def stream_id(self) -> np.ndarray: return self._col("stream_id")

    def aromatic_type_decoded(self) -> np.ndarray:
        if "__arom_dec" in self._agg_cache:
            return self._agg_cache["__arom_dec"]  # type: ignore[return-value]
        codes = self.aromatic_type()
        enum = self._view._arom_enum
        default = self._view._arom_default
        out = np.empty(codes.shape, dtype=object)
        for i, v in enumerate(codes.tolist()):
            out[i] = enum.get(int(v), default)
        self._agg_cache["__arom_dec"] = out
        return out

    def spike_source_decoded(self) -> np.ndarray:
        if "__src_dec" in self._agg_cache:
            return self._agg_cache["__src_dec"]  # type: ignore[return-value]
        codes = self.spike_source()
        enum = self._view._src_enum
        default = self._view._src_default
        out = np.empty(codes.shape, dtype=object)
        for i, v in enumerate(codes.tolist()):
            out[i] = enum.get(int(v), default)
        self._agg_cache["__src_dec"] = out
        return out

    def ccns_phase(self) -> np.ndarray:
        if "__phase_dec" in self._agg_cache:
            return self._agg_cache["__phase_dec"]  # type: ignore[return-value]
        ts = self.timestep()
        v = self._view
        out = np.empty(ts.shape, dtype=object)
        if ts.size:
            out[:] = "cold_return"  # default (ts >= p4)
            out[ts < v._p4] = "cooling"
            out[ts < v._p3] = "warm_hold"
            out[ts < v._p2] = "heating"
            out[ts < v._p1] = "cold_hold"
        self._agg_cache["__phase_dec"] = out
        return out

    # ── Grouped aggregations ────────────────────────────────────────────

    def _int_code_counts(self, codes: np.ndarray, enum: Dict[int, str],
                         default: str) -> Dict[str, int]:
        if codes.size == 0:
            return {}
        vals, counts = np.unique(codes, return_counts=True)
        out: Dict[str, int] = {}
        for v, c in zip(vals.tolist(), counts.tolist()):
            name = enum.get(int(v), default)
            out[name] = out.get(name, 0) + int(c)
        return out

    def type_counts(self) -> Dict[str, int]:
        if "type_counts" in self._agg_cache:
            return dict(self._agg_cache["type_counts"])  # type: ignore[arg-type]
        r = self._int_code_counts(self.aromatic_type(),
                                  self._view._arom_enum,
                                  self._view._arom_default)
        self._agg_cache["type_counts"] = r
        return dict(r)

    def source_counts(self) -> Dict[str, int]:
        if "source_counts" in self._agg_cache:
            return dict(self._agg_cache["source_counts"])  # type: ignore[arg-type]
        r = self._int_code_counts(self.spike_source(),
                                  self._view._src_enum,
                                  self._view._src_default)
        self._agg_cache["source_counts"] = r
        return dict(r)

    def phase_counts(self) -> Dict[str, int]:
        if "phase_counts" in self._agg_cache:
            return dict(self._agg_cache["phase_counts"])  # type: ignore[arg-type]
        if self._n_spikes == 0:
            self._agg_cache["phase_counts"] = {}
            return {}
        ts = self.timestep()
        v = self._view
        counts = [
            int((ts < v._p1).sum()),
            int(((ts >= v._p1) & (ts < v._p2)).sum()),
            int(((ts >= v._p2) & (ts < v._p3)).sum()),
            int(((ts >= v._p3) & (ts < v._p4)).sum()),
            int((ts >= v._p4).sum()),
        ]
        out = {name: c for name, c in zip(_PHASE_ORDER, counts) if c > 0}
        self._agg_cache["phase_counts"] = out
        return dict(out)

    def stream_counts(self) -> Dict[int, int]:
        if "stream_counts" in self._agg_cache:
            return dict(self._agg_cache["stream_counts"])  # type: ignore[arg-type]
        if self._n_spikes == 0:
            self._agg_cache["stream_counts"] = {}
            return {}
        vals, counts = np.unique(self.stream_id(), return_counts=True)
        out = {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}
        self._agg_cache["stream_counts"] = out
        return dict(out)

    def intensity_by_phase(self) -> Dict[str, float]:
        if "int_by_phase" in self._agg_cache:
            return dict(self._agg_cache["int_by_phase"])  # type: ignore[arg-type]
        if self._n_spikes == 0:
            self._agg_cache["int_by_phase"] = {}
            return {}
        phases = self.ccns_phase()
        intensity = self.intensity()
        out: Dict[str, float] = {}
        for name in _PHASE_ORDER:
            m = (phases == name)
            if m.any():
                out[name] = float(intensity[m].sum())
        self._agg_cache["int_by_phase"] = out
        return dict(out)

    def intensity_by_source(self) -> Dict[str, float]:
        if "int_by_src" in self._agg_cache:
            return dict(self._agg_cache["int_by_src"])  # type: ignore[arg-type]
        if self._n_spikes == 0:
            self._agg_cache["int_by_src"] = {}
            return {}
        src_codes = self.spike_source()
        intensity = self.intensity()
        enum = self._view._src_enum
        default = self._view._src_default
        out: Dict[str, float] = {}
        for code in np.unique(src_codes).tolist():
            name = enum.get(int(code), default)
            m = src_codes == code
            out[name] = out.get(name, 0.0) + float(intensity[m].sum())
        self._agg_cache["int_by_src"] = out
        return dict(out)

    def energy_by_phase(self) -> Dict[str, float]:
        if "eng_by_phase" in self._agg_cache:
            return dict(self._agg_cache["eng_by_phase"])  # type: ignore[arg-type]
        if self._n_spikes == 0:
            self._agg_cache["eng_by_phase"] = {}
            return {}
        phases = self.ccns_phase()
        vib = self.vibrational_energy()
        out: Dict[str, float] = {}
        for name in _PHASE_ORDER:
            m = (phases == name)
            if m.any():
                out[name] = float(vib[m].sum())
        self._agg_cache["eng_by_phase"] = out
        return dict(out)

    def water_density_by_phase(self) -> Dict[str, float]:
        if "wd_by_phase" in self._agg_cache:
            return dict(self._agg_cache["wd_by_phase"])  # type: ignore[arg-type]
        if self._n_spikes == 0:
            self._agg_cache["wd_by_phase"] = {}
            return {}
        phases = self.ccns_phase()
        wd = self.water_density()
        out: Dict[str, float] = {}
        for name in _PHASE_ORDER:
            m = (phases == name)
            if m.any():
                out[name] = float(wd[m].mean())
        self._agg_cache["wd_by_phase"] = out
        return dict(out)

    def phase_fraction(self) -> Dict[str, float]:
        pc = self.phase_counts()
        if self._n_spikes == 0 or not pc:
            return {}
        n = float(self._n_spikes)
        return {k: v / n for k, v in pc.items()}

    def source_fraction(self) -> Dict[str, float]:
        sc = self.source_counts()
        if self._n_spikes == 0 or not sc:
            return {}
        n = float(self._n_spikes)
        return {k: v / n for k, v in sc.items()}

    def type_intensity_stats(self) -> Dict[str, Tuple[int, float]]:
        if "type_intensity_stats" in self._agg_cache:
            return dict(self._agg_cache["type_intensity_stats"])  # type: ignore[arg-type]
        if self._n_spikes == 0:
            self._agg_cache["type_intensity_stats"] = {}
            return {}
        codes = self.aromatic_type()
        intensity = self.intensity()
        enum = self._view._arom_enum
        default = self._view._arom_default
        out: Dict[str, Tuple[int, float]] = {}
        for code in np.unique(codes).tolist():
            m = codes == code
            cnt = int(m.sum())
            mean_i = float(intensity[m].mean()) if cnt else 0.0
            name = enum.get(int(code), default)
            if name in out:
                old_cnt, old_mean = out[name]
                new_cnt = old_cnt + cnt
                new_mean = (old_mean * old_cnt + mean_i * cnt) / new_cnt if new_cnt else 0.0
                out[name] = (new_cnt, new_mean)
            else:
                out[name] = (cnt, mean_i)
        self._agg_cache["type_intensity_stats"] = out
        return dict(out)

    # ── Production metrics (response_selectivity) ───────────────────────

    def sharpness(self, centroid: Optional[Tuple[float, float, float]] = None) -> float:
        if self._n_spikes == 0:
            return 0.0
        intensity = self.intensity()
        peak = float(intensity.max()) if intensity.size else 0.0
        if peak <= 0.0:
            return 0.0
        if centroid is None:
            centroid = self.centroid()
        cx, cy, cz = centroid
        x = self.x(); y = self.y(); z = self.z()
        w = intensity.astype(np.float64, copy=False)
        pos = w > 0.0
        if not pos.any():
            return 0.0
        dx = x[pos] - cx
        dy = y[pos] - cy
        dz = z[pos] - cz
        d2 = dx * dx + dy * dy + dz * dz
        wp = w[pos]
        total_w = float(wp.sum())
        if total_w <= 0.0:
            return 0.0
        weighted_d2 = float((wp * d2).sum())
        spread = math.sqrt(weighted_d2 / max(total_w, 1e-12))
        return peak / max(spread, 0.1)

    def temporal_asymmetry(self) -> float:
        if self._n_spikes == 0:
            return 0.0
        if "__tasym" in self._agg_cache:
            return self._agg_cache["__tasym"]  # type: ignore[return-value]
        ts = self.timestep()
        v = self._view
        n_cold = int((ts < v._p1).sum())
        n_warm = int(((ts >= v._p2) & (ts < v._p3)).sum())
        total = n_cold + n_warm
        if total == 0:
            val = 0.0
        else:
            val = abs(n_warm - n_cold) / total
        self._agg_cache["__tasym"] = val
        return val

    def energy_density(self, volume: float) -> float:
        if volume <= 0.0 or self._n_spikes == 0:
            return 0.0
        return float(self.intensity().sum()) / float(volume)

    def contact_coupling(self, contact_changes_per_frame: Optional[Dict[int, int]]) -> float:
        if contact_changes_per_frame is None or len(contact_changes_per_frame) < 3:
            return float("nan")
        frames = sorted(contact_changes_per_frame.keys())
        n = len(frames)
        if n < 3:
            return float("nan")
        frame_col = self.frame_index()
        # Count spikes per frame via np.bincount / isin — vectorized
        if frame_col.size == 0:
            xs = np.zeros(n, dtype=np.float64)
        else:
            max_f = int(frame_col.max())
            counts_by_frame = np.bincount(frame_col.astype(np.int64), minlength=max_f + 1)
            xs = np.array([float(counts_by_frame[f]) if 0 <= f <= max_f else 0.0
                           for f in frames], dtype=np.float64)
        ys = np.array([float(contact_changes_per_frame[f]) for f in frames], dtype=np.float64)
        mx = float(xs.mean())
        my = float(ys.mean())
        cov = float(((xs - mx) * (ys - my)).sum())
        sx = math.sqrt(float(((xs - mx) ** 2).sum()))
        sy = math.sqrt(float(((ys - my) ** 2).sum()))
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return cov / (sx * sy)

    # ── Production metrics (pharmacophore) ──────────────────────────────

    def intensity_weighted_centroid(self) -> Tuple[float, float, float]:
        if "__iwc" in self._agg_cache:
            return self._agg_cache["__iwc"]  # type: ignore[return-value]
        if self._n_spikes == 0:
            val = self.centroid()
        else:
            w = self.intensity().astype(np.float64, copy=False)
            total = float(w.sum())
            if total <= 0:
                val = self.centroid()
            else:
                val = (float((self.x() * w).sum() / total),
                       float((self.y() * w).sum() / total),
                       float((self.z() * w).sum() / total))
        self._agg_cache["__iwc"] = val
        return val

    def voxel_aggregate(self, grid_spacing: float = 2.0) -> Dict[Tuple[int, int, int], dict]:
        if self._n_spikes == 0:
            return {}
        if grid_spacing in self._voxel_cache:
            return self._voxel_cache[grid_spacing]
        try:
            import polars as pl
        except Exception as e:
            raise SiteSpikeViewError(f"polars required for voxel_aggregate: {e}")

        x = self.x(); y = self.y(); z = self.z()
        intensity = self.intensity()
        arom = self.aromatic_type().astype(np.int32, copy=False)
        vib = self.vibrational_energy()
        wd = self.water_density()
        src = self.spike_source().astype(np.int32, copy=False)
        wl = self.wavelength_nm()

        df = pl.DataFrame({
            "x": x, "y": y, "z": z,
            "intensity": intensity,
            "arom": arom,
            "vib": vib,
            "wd": wd,
            "src": src,
            "wl": wl,
        })
        # int() truncation toward zero matches legacy voxelize_spikes: use cast Int32
        df = df.with_columns([
            (pl.col("x") / grid_spacing).cast(pl.Int32).alias("vx"),
            (pl.col("y") / grid_spacing).cast(pl.Int32).alias("vy"),
            (pl.col("z") / grid_spacing).cast(pl.Int32).alias("vz"),
            # Python round() is banker's rounding; polars round() is half-to-even.
            pl.col("wl").round(0).cast(pl.Int32).alias("wl_int"),
        ])

        base = (df.group_by(["vx", "vy", "vz"])
                  .agg([
                      pl.len().alias("count"),
                      pl.sum("intensity").alias("total_intensity"),
                      pl.max("intensity").alias("max_intensity"),
                      pl.mean("x").alias("cx"),
                      pl.mean("y").alias("cy"),
                      pl.mean("z").alias("cz"),
                      pl.sum("vib").alias("vib_sum"),
                      pl.sum("wd").alias("wd_sum"),
                  ]))
        by_type = (df.group_by(["vx", "vy", "vz", "arom"])
                     .agg([pl.len().alias("type_count"),
                           pl.sum("intensity").alias("type_intensity")]))
        by_src = (df.group_by(["vx", "vy", "vz", "src"])
                    .agg([pl.len().alias("src_count")]))
        by_wl = (df.filter(pl.col("wl") > 0)
                   .group_by(["vx", "vy", "vz", "wl_int"])
                   .agg([pl.len().alias("wl_count")]))

        arom_enum = self._view._arom_enum
        arom_default = self._view._arom_default
        src_enum = self._view._src_enum
        src_default = self._view._src_default

        result: Dict[Tuple[int, int, int], dict] = {}
        for row in base.iter_rows(named=True):
            key = (int(row["vx"]), int(row["vy"]), int(row["vz"]))
            count = int(row["count"])
            result[key] = {
                "voxel": key,
                "centroid": [float(row["cx"]), float(row["cy"]), float(row["cz"])],
                "total_intensity": float(row["total_intensity"]),
                "mean_intensity": float(row["total_intensity"]) / count if count else 0.0,
                "max_intensity": float(row["max_intensity"]),
                "spike_count": count,
                "vibrational_energy": float(row["vib_sum"]),
                "mean_water_density": float(row["wd_sum"]) / count if count else 0.0,
                "type_breakdown": {},
                "intensity_by_type": {},
                "sources": {},
                "wavelengths": {},
                "types": {},  # legacy alias for type_breakdown (voxelize_spikes emitted both)
            }
        for row in by_type.iter_rows(named=True):
            key = (int(row["vx"]), int(row["vy"]), int(row["vz"]))
            tname = arom_enum.get(int(row["arom"]), arom_default)
            v = result[key]
            v["type_breakdown"][tname] = v["type_breakdown"].get(tname, 0) + int(row["type_count"])
            v["types"][tname] = v["types"].get(tname, 0) + int(row["type_count"])
            v["intensity_by_type"][tname] = v["intensity_by_type"].get(tname, 0.0) + float(row["type_intensity"])
        for row in by_src.iter_rows(named=True):
            key = (int(row["vx"]), int(row["vy"]), int(row["vz"]))
            sname = src_enum.get(int(row["src"]), src_default)
            v = result[key]
            v["sources"][sname] = v["sources"].get(sname, 0) + int(row["src_count"])
        for row in by_wl.iter_rows(named=True):
            key = (int(row["vx"]), int(row["vy"]), int(row["vz"]))
            v = result[key]
            wl_int = int(row["wl_int"])
            v["wavelengths"][wl_int] = v["wavelengths"].get(wl_int, 0) + int(row["wl_count"])
        for v in result.values():
            if v["intensity_by_type"]:
                v["dominant_type"] = max(v["intensity_by_type"], key=v["intensity_by_type"].get)
            else:
                v["dominant_type"] = arom_default

        self._voxel_cache[grid_spacing] = result
        return result

    # ── Wave-1 contract extensions (added in Step A) ────────────────────

    def to_polars(self, columns: Optional[List[str]] = None):
        """Return a polars DataFrame of the site's rows.

        Args:
            columns: list of column names to include. If None, include all 14
                     Arrow columns plus decoded 'ccns_phase'.
                     Valid names: any of the 14 Arrow columns (x, y, z,
                     intensity, aromatic_type, wavelength_nm, spike_source,
                     aromatic_residue_id, water_density, vibrational_energy,
                     n_nearby_excited, timestep, frame_index, stream_id) plus
                     'ccns_phase' (decoded string), 'aromatic_type_decoded',
                     'spike_source_decoded'.

        Columns are materialized via the view-level full-column cache + per-slice
        integer-gather (O(n_site) per column). Decoded columns are materialized
        via the view's enum tables.

        Hot-path safe: yes (no per-spike dict construction).
        """
        try:
            import polars as pl
        except Exception as e:
            raise SiteSpikeViewError(f"polars required for to_polars: {e}")
        RAW_COLS = list(_REQUIRED_ARROW_COLUMNS)
        DECODED = {"ccns_phase", "aromatic_type_decoded", "spike_source_decoded"}
        if columns is None:
            columns = RAW_COLS + ["ccns_phase"]
        data = {}
        for name in columns:
            if name in RAW_COLS:
                data[name] = self._col(name)
            elif name == "ccns_phase":
                data[name] = self.ccns_phase()
            elif name == "aromatic_type_decoded":
                data[name] = self.aromatic_type_decoded()
            elif name == "spike_source_decoded":
                data[name] = self.spike_source_decoded()
            else:
                raise SiteSpikeViewError(f"unknown column: {name}")
        return pl.DataFrame(data)

    def temporal_windows_18dim(self, n_windows: int = 32,
                                max_events_cap: Optional[int] = None) -> np.ndarray:
        """Return a (n_windows, 18) float32 matrix matching
        `train_v006.compute_window_features` exactly.

        Optional max_events_cap preserves the legacy subsample behavior of
        `train_v006.parse_spike_events`: if n_spikes > max_events_cap, take
        every `n_spikes // max_events_cap`-th spike, truncate to max_events_cap
        rows. Default None = no cap (Step A contract default).

        Window feature layout (18 dims):
          0:  spike_count
          1:  mean_intensity
          2:  peak_intensity
          3:  std_intensity
          4:  burst_count           (n_nearby_excited >= 3)
          5:  isi_mean              (inter-spike interval mean; timestep-based)
          6:  isi_std
          7:  ch_uv_fraction        (lowercased spike_source contains 'uv')
          8:  ch_lif_fraction       (contains 'lif')
          9:  ch_efp_fraction       (contains 'efp')
          10: ph_cold_hold_fraction
          11: ph_heating_fraction
          12: ph_warm_hold_fraction
          13: ph_cooling_fraction
          14: ph_cold_return_fraction
          15: mean_n_nearby_excited
          16: mean_vibrational_energy
          17: mean_water_density

        Parity target: scripts/training/train_v006.compute_window_features.
        Tolerance: ≤1e-5 relative on float fields; exact on counts.
        Hot-path safe: yes — fully vectorized.
        """
        features = np.zeros((n_windows, 18), dtype=np.float32)
        if self._n_spikes == 0:
            return features

        # Optional legacy cap: train_v006.parse_spike_events subsample step.
        subsample_idx = None
        if max_events_cap is not None and self._n_spikes > max_events_cap:
            step = self._n_spikes // max_events_cap
            subsample_idx = np.arange(0, self._n_spikes, step)[:max_events_cap]

        ts = self.timestep().astype(np.int64, copy=False)
        intensity = self.intensity().astype(np.float32, copy=False)
        n_excited = self.n_nearby_excited().astype(np.int32, copy=False)
        vib = self.vibrational_energy().astype(np.float32, copy=False)
        wd = self.water_density().astype(np.float32, copy=False)
        src_dec = self.spike_source_decoded()
        phase_dec = self.ccns_phase()
        if subsample_idx is not None:
            ts = ts[subsample_idx]
            intensity = intensity[subsample_idx]
            n_excited = n_excited[subsample_idx]
            vib = vib[subsample_idx]
            wd = wd[subsample_idx]
            src_dec = src_dec[subsample_idx]
            phase_dec = phase_dec[subsample_idx]

        if ts.size == 0:
            return features
        t_min = int(ts.min())
        t_max = int(ts.max())
        if t_max <= t_min:
            t_max = t_min + 1
        window_size = (t_max - t_min) / n_windows
        # Exact legacy formula: w = min(int((t - t_min) / window_size), n_windows - 1)
        w_idx = ((ts - t_min) / window_size).astype(np.int64)
        np.minimum(w_idx, n_windows - 1, out=w_idx)

        # dim 0: count per window
        counts = np.bincount(w_idx, minlength=n_windows)
        features[:, 0] = counts.astype(np.float32)

        # dims 1-3: mean, max, std of intensity per window
        sum_i = np.bincount(w_idx, weights=intensity.astype(np.float64), minlength=n_windows)
        sum_i2 = np.bincount(w_idx, weights=(intensity.astype(np.float64) ** 2), minlength=n_windows)
        # peak via np.maximum.reduceat requires sorted — use a loop-free trick:
        # put intensity into a (n_windows,) max using np.maximum.at (scatter-max)
        peak = np.full(n_windows, -np.inf, dtype=np.float32)
        np.maximum.at(peak, w_idx, intensity)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_i = np.where(counts > 0, sum_i / np.maximum(counts, 1), 0.0)
            # Legacy uses np.std with default ddof=0 (population variance).
            var_i = np.where(counts > 0, sum_i2 / np.maximum(counts, 1) - mean_i ** 2, 0.0)
            # numerical floor at 0 (can go slightly negative from rounding)
            var_i = np.clip(var_i, 0.0, None)
            std_i = np.sqrt(var_i)
            # Legacy emits std only when len(intensities) > 1
            std_i = np.where(counts > 1, std_i, 0.0)
        features[:, 1] = mean_i.astype(np.float32)
        features[:, 2] = np.where(counts > 0, peak, 0.0).astype(np.float32)
        features[:, 3] = std_i.astype(np.float32)

        # dim 4: burst_count = sum(n_nearby_excited >= 3)
        burst_mask = (n_excited >= 3)
        features[:, 4] = np.bincount(w_idx[burst_mask], minlength=n_windows).astype(np.float32)

        # dims 5-6: ISI mean + std per window (from sorted timesteps per window)
        # Legacy: for each window, sort timesteps ascending, take diffs where diff>0,
        # compute mean and std (ddof=0) over positive diffs.
        # Vectorized via per-window groupby-sort:
        order = np.lexsort((ts, w_idx))  # sort by window then by timestep
        ts_sorted = ts[order]
        w_sorted = w_idx[order]
        # boundary[i] = True when w_sorted[i] != w_sorted[i-1]
        # ISI within-window only
        if ts_sorted.size >= 2:
            diffs = ts_sorted[1:] - ts_sorted[:-1]
            same_window = (w_sorted[1:] == w_sorted[:-1])
            valid = same_window & (diffs > 0)
            valid_diffs = diffs[valid].astype(np.float64)
            valid_windows = w_sorted[1:][valid]
            if valid_diffs.size:
                sum_isi = np.bincount(valid_windows, weights=valid_diffs, minlength=n_windows)
                sum_isi2 = np.bincount(valid_windows, weights=valid_diffs ** 2, minlength=n_windows)
                cnt_isi = np.bincount(valid_windows, minlength=n_windows)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_isi = np.where(cnt_isi > 0, sum_isi / np.maximum(cnt_isi, 1), 0.0)
                    var_isi = np.where(cnt_isi > 0, sum_isi2 / np.maximum(cnt_isi, 1) - mean_isi ** 2, 0.0)
                    var_isi = np.clip(var_isi, 0.0, None)
                    std_isi = np.sqrt(var_isi)
                    std_isi = np.where(cnt_isi > 1, std_isi, 0.0)
                features[:, 5] = mean_isi.astype(np.float32)
                features[:, 6] = std_isi.astype(np.float32)

        # dims 7-9: channel fractions (uv/lif/efp substring match on lowercased source)
        CHANNEL_MAP = {"uv": 7, "lif": 8, "efp": 9}
        src_lower = np.char.lower(src_dec.astype(str))
        counts_nz = np.maximum(counts, 1).astype(np.float64)
        for ch_name, ch_idx in CHANNEL_MAP.items():
            ch_hit = np.char.find(src_lower, ch_name) >= 0
            ch_counts = np.bincount(w_idx[ch_hit], minlength=n_windows).astype(np.float64)
            frac = ch_counts / counts_nz
            features[:, ch_idx] = np.where(counts > 0, frac, 0.0).astype(np.float32)

        # dims 10-14: phase fractions (exact string match after lowercase)
        PHASE_MAP = {
            "cold_hold": 10, "heating": 11, "warm_hold": 12,
            "cooling": 13, "cold_return": 14,
        }
        phase_lower = np.char.lower(phase_dec.astype(str))
        for ph_name, ph_idx in PHASE_MAP.items():
            ph_hit = np.char.find(phase_lower, ph_name) >= 0
            ph_counts = np.bincount(w_idx[ph_hit], minlength=n_windows).astype(np.float64)
            frac = ph_counts / counts_nz
            features[:, ph_idx] = np.where(counts > 0, frac, 0.0).astype(np.float32)

        # dims 15-17: context means
        sum_nex = np.bincount(w_idx, weights=n_excited.astype(np.float64), minlength=n_windows)
        sum_vib = np.bincount(w_idx, weights=vib.astype(np.float64), minlength=n_windows)
        sum_wd = np.bincount(w_idx, weights=wd.astype(np.float64), minlength=n_windows)
        with np.errstate(invalid="ignore", divide="ignore"):
            features[:, 15] = np.where(counts > 0, sum_nex / counts_nz, 0.0).astype(np.float32)
            features[:, 16] = np.where(counts > 0, sum_vib / counts_nz, 0.0).astype(np.float32)
            features[:, 17] = np.where(counts > 0, sum_wd / counts_nz, 0.0).astype(np.float32)
        return features

    def feature_channels(
        self,
        residue_ids: np.ndarray,
        source_filter: Optional[str] = None,
        phase_filter: Optional[str] = None,
    ) -> np.ndarray:
        """For each residue_id in input, compute per-source per-phase aggregate
        stats (count, mean_intensity, mean_water_density, mean_vibrational_energy,
        mean_n_nearby_excited, mean_wavelength_nm) over spikes filtered by:
          - aromatic_residue_id == rid
          - (if source_filter set) decoded spike_source == source_filter
          - (if phase_filter set) decoded ccns_phase == phase_filter

        Args:
            residue_ids: numpy int array of residue ids to query.
            source_filter: decoded spike source name (e.g. "UV") or None.
            phase_filter: decoded phase name (e.g. "warm_hold") or None.

        Returns:
            np.ndarray[float32, shape=(len(residue_ids), 6)]:
              [:,0] count
              [:,1] mean_intensity
              [:,2] mean_water_density
              [:,3] mean_vibrational_energy
              [:,4] mean_n_nearby_excited
              [:,5] mean_wavelength_nm

        Empty filter result → row of zeros for that residue.
        Tolerance: ≤1e-6 relative on means; exact on counts.
        Hot-path safe: yes.
        """
        residue_ids = np.asarray(residue_ids, dtype=np.int64)
        out = np.zeros((residue_ids.size, 6), dtype=np.float32)
        if self._n_spikes == 0 or residue_ids.size == 0:
            return out
        arid = self.aromatic_residue_id().astype(np.int64, copy=False)
        intensity = self.intensity().astype(np.float64, copy=False)
        wd = self.water_density().astype(np.float64, copy=False)
        vib = self.vibrational_energy().astype(np.float64, copy=False)
        nex = self.n_nearby_excited().astype(np.float64, copy=False)
        wl = self.wavelength_nm().astype(np.float64, copy=False)
        base_mask = np.ones(self._n_spikes, dtype=bool)
        if source_filter is not None:
            base_mask &= (self.spike_source_decoded() == source_filter)
        if phase_filter is not None:
            base_mask &= (self.ccns_phase() == phase_filter)
        for i, rid in enumerate(residue_ids.tolist()):
            m = base_mask & (arid == int(rid))
            n = int(m.sum())
            if n == 0:
                continue
            out[i, 0] = float(n)
            out[i, 1] = float(intensity[m].mean())
            out[i, 2] = float(wd[m].mean())
            out[i, 3] = float(vib[m].mean())
            out[i, 4] = float(nex[m].mean())
            out[i, 5] = float(wl[m].mean())
        return out

    def min_dist_to_point(self, p: Tuple[float, float, float]) -> float:
        """Minimum Euclidean distance from any spike in this site to point p.
        Returns math.inf if site is empty.

        Tolerance: ≤1e-6 relative.
        Hot-path safe: yes (pure numpy vectorized).
        """
        if self._n_spikes == 0:
            return float("inf")
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        dx = self.x().astype(np.float64, copy=False) - px
        dy = self.y().astype(np.float64, copy=False) - py
        dz = self.z().astype(np.float64, copy=False) - pz
        d2 = dx * dx + dy * dy + dz * dz
        return float(np.sqrt(d2.min()))

    # ── Debug / non-hot-path ────────────────────────────────────────────

    def to_legacy_header_dict(self) -> dict:
        return {
            "site_id": self._sid,
            "centroid": list(self.centroid()),
            "n_spikes": self._n_spikes,
            "lining_cutoff": self._view._lining_cutoff,
            "open_frequency": self.open_frequency(),
        }

    def to_legacy_spikes_debug_sample(self, n: int = 100) -> List[dict]:
        import warnings
        if n > 10000:
            warnings.warn("to_legacy_spikes_debug_sample is non-hot-path only", DeprecationWarning)
        if self._n_spikes == 0:
            return []
        upper = min(n, self._n_spikes)
        return self.debug_row_sample(list(range(upper)))

    def debug_row_sample(self, indices: Optional[List[int]] = None) -> List[dict]:
        if self._n_spikes == 0:
            return []
        if indices is None:
            indices = list(range(self._n_spikes))
        if not indices:
            return []
        x = self.x(); y = self.y(); z = self.z()
        intensity = self.intensity()
        arom = self.aromatic_type_decoded()
        wl = self.wavelength_nm()
        src = self.spike_source_decoded()
        arid = self.aromatic_residue_id()
        wd = self.water_density()
        vib = self.vibrational_energy()
        nex = self.n_nearby_excited()
        ts = self.timestep()
        fi = self.frame_index()
        sid = self.stream_id()
        phase = self.ccns_phase()
        out = []
        for i in indices:
            out.append({
                "x": float(x[i]), "y": float(y[i]), "z": float(z[i]),
                "intensity": float(intensity[i]),
                "type": str(arom[i]),
                "wavelength_nm": float(wl[i]),
                "spike_source": str(src[i]),
                "aromatic_residue_id": int(arid[i]),
                "water_density": float(wd[i]),
                "vibrational_energy": float(vib[i]),
                "n_nearby_excited": int(nex[i]),
                "timestep": int(ts[i]),
                "frame_index": int(fi[i]),
                "ccns_phase": str(phase[i]),
                "stream_id": int(sid[i]),
            })
        return out
