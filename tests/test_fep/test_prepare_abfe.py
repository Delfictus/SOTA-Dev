"""Tests for prepare_abfe.py — ABFE protocol configuration."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from scripts.fep.prepare_abfe import (
    ABFEProtocol,
    LambdaSchedule,
    _make_complex_schedule,
    _make_solvent_schedule,
    prepare_abfe,
    main,
)


class TestLambdaSchedules:
    def test_complex_schedule_default(self):
        sched = _make_complex_schedule()
        assert sched.n_windows == 42
        assert sched.name == "complex"
        assert len(sched.lambda_electrostatics) == 42
        assert len(sched.lambda_sterics) == 42
        assert len(sched.lambda_restraints) == 42

    def test_complex_schedule_endpoints(self):
        sched = _make_complex_schedule()
        # First window: elec=1 (fully coupled), sterics=1, restraint=0
        assert sched.lambda_electrostatics[0] == 1.0
        assert sched.lambda_sterics[0] == 1.0
        assert sched.lambda_restraints[0] == 0.0
        # Last window: elec=0, sterics=0 (fully decoupled), restraint=1
        assert sched.lambda_electrostatics[-1] == 0.0
        assert sched.lambda_sterics[-1] == 0.0
        assert sched.lambda_restraints[-1] == 1.0

    def test_solvent_schedule_default(self):
        sched = _make_solvent_schedule()
        assert sched.n_windows == 31
        assert sched.name == "solvent"
        # No restraints in solvent leg
        assert all(r == 0.0 for r in sched.lambda_restraints)

    def test_solvent_schedule_endpoints(self):
        sched = _make_solvent_schedule()
        assert sched.lambda_electrostatics[0] == 1.0
        assert sched.lambda_sterics[0] == 1.0
        assert sched.lambda_electrostatics[-1] == 0.0
        assert sched.lambda_sterics[-1] == 0.0

    def test_custom_window_count(self):
        sched = _make_complex_schedule(n_windows=30)
        assert sched.n_windows == 30
        assert len(sched.lambda_electrostatics) == 30

    def test_to_dict(self):
        sched = _make_complex_schedule()
        d = sched.to_dict()
        assert d["n_windows"] == 42
        assert "lambda_electrostatics" in d


class TestABFEProtocol:
    def test_total_windows(self):
        protocol = ABFEProtocol(
            compound_id="test",
            complex_schedule=_make_complex_schedule(),
            solvent_schedule=_make_solvent_schedule(),
        )
        assert protocol.total_windows == 73

    def test_total_simulation_time(self):
        protocol = ABFEProtocol(
            compound_id="test",
            complex_schedule=_make_complex_schedule(),
            solvent_schedule=_make_solvent_schedule(),
            simulation_time_ns=5.0,
            n_repeats=3,
        )
        # 73 windows * 5 ns * 3 repeats = 1095 ns
        assert protocol.total_simulation_time_ns == 1095.0

    def test_to_json(self):
        protocol = ABFEProtocol(
            compound_id="test",
            complex_schedule=_make_complex_schedule(),
            solvent_schedule=_make_solvent_schedule(),
        )
        j = protocol.to_json()
        d = json.loads(j)
        assert d["compound_id"] == "test"
        assert d["total_windows"] == 73

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = ABFEProtocol(
                compound_id="save_test",
                complex_schedule=_make_complex_schedule(),
                solvent_schedule=_make_solvent_schedule(),
                output_dir=tmpdir,
            )
            path = protocol.save()
            assert os.path.isfile(path)
            with open(path) as f:
                d = json.load(f)
            assert d["compound_id"] == "save_test"


class TestPrepareABFE:
    def test_full_prepare(self, mock_protein_pdb, mock_ligand_sdf):
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = prepare_abfe(
                compound_id="ABFE_001",
                protein_pdb=mock_protein_pdb,
                ligand_sdf=mock_ligand_sdf,
                output_dir=tmpdir,
            )
            assert protocol.compound_id == "ABFE_001"
            assert protocol.total_windows == 73
            assert protocol.n_repeats == 3
            # Protocol JSON should be saved
            expected = os.path.join(tmpdir, "ABFE_001_abfe_protocol.json")
            assert os.path.isfile(expected)

    def test_custom_parameters(self, mock_protein_pdb, mock_ligand_sdf):
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = prepare_abfe(
                compound_id="CUSTOM",
                protein_pdb=mock_protein_pdb,
                ligand_sdf=mock_ligand_sdf,
                output_dir=tmpdir,
                n_complex_windows=30,
                n_solvent_windows=20,
                simulation_time_ns=2.0,
                n_repeats=2,
                use_rest2=True,
            )
            assert protocol.total_windows == 50
            assert protocol.simulation_time_ns == 2.0
            assert protocol.use_rest2 is True
            # 50 * 2 * 2 = 200 ns
            assert protocol.total_simulation_time_ns == 200.0


class TestCLI:
    def test_main(self, mock_protein_pdb, mock_ligand_sdf):
        with tempfile.TemporaryDirectory() as tmpdir:
            main([
                "--protein", mock_protein_pdb,
                "--ligand", mock_ligand_sdf,
                "--output-dir", tmpdir,
                "--compound-id", "CLI_TEST",
            ])
            expected = os.path.join(tmpdir, "CLI_TEST_abfe_protocol.json")
            assert os.path.isfile(expected)
