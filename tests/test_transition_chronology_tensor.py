from pathlib import Path

import polars as pl


TENSOR = Path(
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/transition_chronology_tensor.parquet"
)


def test_chronology_tensor_has_required_runtime_columns() -> None:
    frame = pl.read_parquet(TENSOR)
    assert frame.height > 0
    for column in ("true_md_step", "event_type", "temporal_overlap_entropy", "replica", "stream"):
        assert column in frame.columns
    assert frame.get_column("true_md_step").null_count() == 0
    assert frame.filter(pl.col("residues").list.len() > 0).height > 0
    assert frame.filter(pl.col("voxel_idx").is_not_null()).height > 0
    assert "DIRECT_PATHWAY_NODE" in set(frame.get_column("localization_status").to_list())


def test_chronology_tensor_rejects_missing_true_md_step_contract() -> None:
    frame = pl.read_parquet(TENSOR, n_rows=5)
    assert "true_md_step" in frame.columns
