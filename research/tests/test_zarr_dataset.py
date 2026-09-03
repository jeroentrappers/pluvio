"""model.zarr_dataset.ZarrCorrectionDataset against the synthetic store."""

from __future__ import annotations

import numpy as np
from model.zarr_dataset import ZarrCorrectionDataset

from tests._store_spec import GRID_N, LEADS_MIN, NAN_ISSUE_IDX


def test_grid_hw_from_store(synthetic_store):
    ds = ZarrCorrectionDataset(synthetic_store, leads_min=tuple(lead for lead in LEADS_MIN if lead))
    assert ds.grid_hw == (GRID_N, GRID_N)


def test_build_input_shape_and_static_discovery(synthetic_store):
    ds = ZarrCorrectionDataset(synthetic_store, leads_min=tuple(lead for lead in LEADS_MIN if lead))

    assert "static_elevation_m" in ds.static_channels
    assert set(ds.aux_channels) == {"msg_ir108", "alaro_precip"}

    expected_channels = ds.history_steps + 4 + len(ds.aux_channels) + len(ds.static_channels)
    assert ds.n_channels == expected_channels

    s = ds.index[0]
    chans = ds.build_input(s.issue_idx, s.lead_min, s.history_idx)
    assert chans.shape == (expected_channels, GRID_N, GRID_N)


def test_index_target_mapping(synthetic_store):
    ds = ZarrCorrectionDataset(synthetic_store, leads_min=tuple(lead for lead in LEADS_MIN if lead))
    assert len(ds.index) > 0

    for s in ds.index:
        issue_epoch = int(ds._issue_epoch[s.issue_idx])
        target_epoch = int(ds._issue_epoch[s.target_idx])
        assert target_epoch == issue_epoch + s.lead_min * 60


def test_nan_truth_issue_excluded_as_target_but_usable_as_history(synthetic_store):
    ds = ZarrCorrectionDataset(synthetic_store)

    # All-NaN truth at NAN_ISSUE_IDX must never be picked as a sample target.
    assert NAN_ISSUE_IDX not in {s.target_idx for s in ds.index}

    # It can still appear in another sample's radar HISTORY (only its truth
    # is unusable, not its radar) — build_input must come back all-finite
    # despite the NaN patch in its own radar field.
    sample = next(s for s in ds.index if NAN_ISSUE_IDX in s.history_idx)
    chans = ds.build_input(sample.issue_idx, sample.lead_min, sample.history_idx)
    assert np.isfinite(chans).all()
