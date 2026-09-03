"""model.zarr_dataset.ZarrCorrectionDataset against the synthetic store."""

from __future__ import annotations

from model.zarr_dataset import ZarrCorrectionDataset
from tests.conftest import GRID_N, LEADS_MIN


def test_grid_hw_from_store(synthetic_store):
    ds = ZarrCorrectionDataset(synthetic_store, leads_min=tuple(l for l in LEADS_MIN if l))
    assert ds.grid_hw == (GRID_N, GRID_N)


def test_build_input_shape_and_static_discovery(synthetic_store):
    ds = ZarrCorrectionDataset(synthetic_store, leads_min=tuple(l for l in LEADS_MIN if l))

    assert "static_elevation_m" in ds.static_channels
    assert set(ds.aux_channels) == {"msg_ir108", "alaro_precip"}

    expected_channels = ds.history_steps + 4 + len(ds.aux_channels) + len(ds.static_channels)
    assert ds.n_channels == expected_channels

    s = ds.index[0]
    chans = ds.build_input(s.issue_idx, s.lead_min, s.history_idx)
    assert chans.shape == (expected_channels, GRID_N, GRID_N)


def test_index_target_mapping(synthetic_store):
    ds = ZarrCorrectionDataset(synthetic_store, leads_min=tuple(l for l in LEADS_MIN if l))
    assert len(ds.index) > 0

    for s in ds.index:
        issue_epoch = int(ds._issue_epoch[s.issue_idx])
        target_epoch = int(ds._issue_epoch[s.target_idx])
        assert target_epoch == issue_epoch + s.lead_min * 60
