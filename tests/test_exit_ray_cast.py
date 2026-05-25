from __future__ import annotations

import numpy as np

from prism_dstw.scoring.exit_atom_ray_cast import ExitAtomRayCaster


class MockLookup:
    def __init__(self, classification: str | None) -> None:
        self.classification = classification
        self.grid = {0: {"classification": classification}} if classification is not None else {}

    def xyz_to_voxel(self, xyz: np.ndarray) -> int | None:
        if self.classification is None:
            return None
        return 0


def test_blocked_exit_gets_neginf() -> None:
    ray_caster = ExitAtomRayCaster(MockLookup("stable_occupied"))
    masks = ray_caster.compute_exit_masks(
        exit_atom_positions=np.array([[5.0, 5.0, 5.0]], dtype=np.float32),
        molecule_centroid=np.array([4.0, 5.0, 5.0], dtype=np.float32),
    )
    assert masks[0].item() == float("-inf")


def test_open_exit_gets_zero() -> None:
    ray_caster = ExitAtomRayCaster(MockLookup("thermally_activated"))
    masks = ray_caster.compute_exit_masks(
        exit_atom_positions=np.array([[5.0, 5.0, 5.0]], dtype=np.float32),
        molecule_centroid=np.array([4.0, 5.0, 5.0], dtype=np.float32),
    )
    assert float(masks[0].item()) == 0.0


def test_solvent_exit_gets_mild_penalty() -> None:
    ray_caster = ExitAtomRayCaster(MockLookup(None))
    masks = ray_caster.compute_exit_masks(
        exit_atom_positions=np.array([[9999.0, 0.0, 0.0]], dtype=np.float32),
        molecule_centroid=np.array([9998.0, 0.0, 0.0], dtype=np.float32),
    )
    assert abs(float(masks[0].item()) - (-2.0)) < 0.01
