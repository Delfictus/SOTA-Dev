"""Tests for PipelineConfig interface."""
from __future__ import annotations

import json
import os
import pickle
import tempfile

import pytest

from scripts.interfaces.pipeline_config import (
    DockingConfig,
    FEPConfig,
    FilterConfig,
    PipelineConfig,
)


class TestSubConfigs:
    def test_docking_defaults(self):
        dc = DockingConfig()
        assert dc.engine == "unidock+gnina"
        assert dc.exhaustiveness == 32
        assert dc.num_modes == 20
        assert dc.box_padding == 4.0
        assert dc.max_box_side == 40.0

    def test_filter_defaults(self):
        fc = FilterConfig()
        assert fc.qed_min == 0.3
        assert fc.sa_max == 6.0
        assert fc.lipinski_max_violations == 1
        assert fc.pains_reject is True
        assert fc.tanimoto_max == 0.85

    def test_fep_defaults(self):
        fc = FEPConfig()
        assert fc.method == "ABFE"
        assert fc.n_repeats == 3
        assert fc.n_lambda_windows == 20
        assert fc.simulation_time_ns == 5.0


class TestPipelineConfig:
    def test_construction(self, sample_config):
        pc = sample_config
        assert pc.project_name == "KRAS-G12C Screen"
        assert pc.target_name == "KRAS_G12C"
        assert pc.random_seed == 42
        assert isinstance(pc.docking, DockingConfig)
        assert isinstance(pc.filtering, FilterConfig)
        assert isinstance(pc.fep, FEPConfig)

    def test_default_stages(self, sample_config):
        assert "pharmacophore" in sample_config.stages_enabled
        assert "docking" in sample_config.stages_enabled
        assert "fep" in sample_config.stages_enabled

    def test_json_round_trip(self, sample_config):
        j = sample_config.to_json()
        pc2 = PipelineConfig.from_json(j)
        assert pc2.project_name == "KRAS-G12C Screen"
        assert pc2.docking.engine == "unidock+gnina"
        assert pc2.filtering.qed_min == 0.3
        assert pc2.fep.method == "ABFE"

    def test_nested_config_override(self):
        pc = PipelineConfig(
            project_name="Custom",
            target_name="TP53",
            pdb_id="1TSR",
            receptor_pdb="/r.pdb",
            topology_json="/t.json",
            binding_sites_json="/s.json",
            output_dir="/out",
            docking=DockingConfig(engine="gnina", exhaustiveness=64),
            filtering=FilterConfig(qed_min=0.5),
            fep=FEPConfig(method="RBFE", n_repeats=5),
        )
        assert pc.docking.engine == "gnina"
        assert pc.docking.exhaustiveness == 64
        assert pc.filtering.qed_min == 0.5
        assert pc.fep.method == "RBFE"
        assert pc.fep.n_repeats == 5

    def test_json_file_round_trip(self, sample_config):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name
        try:
            sample_config.save_json(tmp_path)
            pc2 = PipelineConfig.from_json_file(tmp_path)
            assert pc2.target_name == sample_config.target_name
            assert pc2.docking.engine == sample_config.docking.engine
        finally:
            os.unlink(tmp_path)

    def test_dict_round_trip(self, sample_config):
        d = sample_config.to_dict()
        pc2 = PipelineConfig.from_dict(d)
        assert pc2.to_dict() == d

    def test_pickle_round_trip(self, sample_config):
        data = sample_config.to_pickle()
        pc2 = PipelineConfig.from_pickle(data)
        assert pc2.target_name == "KRAS_G12C"

    def test_pickle_type_check(self):
        bad = pickle.dumps(None)
        with pytest.raises(TypeError, match="Expected PipelineConfig"):
            PipelineConfig.from_pickle(bad)

    def test_from_dict_missing_subconfigs(self):
        """from_dict should work even when sub-config keys are absent."""
        d = {
            "project_name": "Minimal",
            "target_name": "X",
            "pdb_id": "XXXX",
            "receptor_pdb": "/r.pdb",
            "topology_json": "/t.json",
            "binding_sites_json": "/s.json",
            "output_dir": "/out",
        }
        pc = PipelineConfig.from_dict(d)
        assert pc.docking.engine == "unidock+gnina"
        assert pc.filtering.qed_min == 0.3
