"""4.1: composite frames resampled onto the model grid, KNMI leads time-shifted,
aux carried forward — and the assembled input has the dataset's layout."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from model.zarr_dataset import ZarrCorrectionDataset
from tools import lowlatency_infer as ll
from tools.scoreboard import QpeTruth


def _qpe_day(root, day: dt.date, bounds, value_at_slot):
    """Tiny QPE day store: rate (288, 30, 30) with the given per-slot constant."""
    p = root / f"{day:%Y/%m}" / f"{day:%d}.zarr"
    p.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(p), mode="w")
    rate = np.full((288, 30, 30), np.nan, dtype="float32")
    for slot, v in value_at_slot.items():
        rate[slot] = v
    g.create_array("rate", data=rate, chunks=(12, 30, 30))
    g.attrs.update({"bounds": list(bounds), "bounds_convention": "outer_edges",
                    "grid_row_order": "north_to_south"})
    return p


def test_sample_regular_raster_is_exact_on_constant_and_zero_outside():
    rate = np.full((30, 30), 3.0, dtype="float32")
    lat = np.array([[52.0, 55.0]]); lon = np.array([[4.0, 4.0]])   # 55 N is outside the 49-54 box
    out = ll.sample_regular_raster(rate, (1.0, 49.0, 8.0, 54.0), lat, lon)
    assert out[0, 0] == pytest.approx(3.0) and out[0, 1] == 0.0


def test_shifted_lead_index_adds_the_issue_age_and_clamps():
    leads = [0, 30, 60, 90]
    assert ll.shifted_lead_index(leads, 35.0, 30) == 2      # 65 -> nearest 60
    assert ll.shifted_lead_index(leads, 5.0, 30) == 1
    assert ll.shifted_lead_index(leads, 50.0, 90) == 3      # 140 -> clamped to 90


def test_lowlatency_input_matches_dataset_layout_and_uses_composite_history(synthetic_store, tmp_path):
    ds = ZarrCorrectionDataset(synthetic_store, leads_min=(30, 60), build_index=False)
    root = zarr.open_group(str(synthetic_store), mode="r")
    _grid, lat, lon = ll.model_grid_latlon(root)
    epochs = ds._issue_epoch
    t = (int(epochs[10]) // 300 + 2) * 300                    # first 5-min slots after issue 10 (6 to 10 min later)
    slots = {}
    for k in range(ds.history_steps):
        te = t - (ds.history_steps - 1 - k) * ds.history_step_min * 60
        d = dt.datetime.fromtimestamp(te, dt.UTC)
        slots.setdefault(d.date(), {})[(d.hour * 60 + d.minute) // 5] = 1.0 + k   # oldest 1.0 … newest H
    qroot = tmp_path / "qpe"
    for d, sv in slots.items():
        _qpe_day(qroot, d, (0.0, 48.0, 9.0, 55.0), sv)      # covers the whole synthetic box
    qpe = QpeTruth(qroot)
    x, info = ll.build_lowlatency_input(ds, root, qpe, t, 30, lat, lon, ds.aux_channels)
    assert x.shape == (ds.n_channels, *ds.grid_hw)
    for k in range(ds.history_steps):
        assert np.allclose(x[k], 1.0 + k)                      # composite frames, oldest→newest
    assert info["issue_idx"] == 10 and 5.0 <= info["age_min"] <= 10.0
    assert info["knmi_lead_used"] == 30                        # 10 + 30 = 40 -> nearest 30
    ref = ds.build_input(10, 30, ds.history_for(10))
    H = ds.history_steps
    assert x[H + 1] == pytest.approx(ref[H + 1])               # lead plane identical
    np.testing.assert_array_equal(x[H + 4:], ref[H + 4:])      # aux + statics carried forward verbatim


def test_evaluate_day_scores_both_paths_on_the_same_valid_times(synthetic_store, tmp_path):
    import torch

    ds = ZarrCorrectionDataset(synthetic_store, leads_min=(30, 60, 90), build_index=False)
    root = zarr.open_group(str(synthetic_store), mode="r")
    _grid, lat, lon = ll.model_grid_latlon(root)
    epochs = ds._issue_epoch
    day = dt.datetime.fromtimestamp(int(epochs[12]), dt.UTC).date()
    qroot = tmp_path / "qpe"
    for d in (day - dt.timedelta(days=1), day, day + dt.timedelta(days=1)):
        _qpe_day(qroot, d, (0.0, 48.0, 9.0, 55.0), {s: 2.0 for s in range(288)})   # wet everywhere, always
    qpe = QpeTruth(qroot)

    class Two(torch.nn.Module):                        # predicts 2 mm/h everywhere → perfect vs the truth
        def forward(self, x):
            return torch.full((x.shape[0], 1, *x.shape[-2:]), 2.0)

    res = ll.evaluate_day(ds, root, qpe, Two(), day, [30, 60], lat, lon, ds.aux_channels,
                          wallclock_lag_min=0.0, step_min=5)
    for L in ("30", "60"):
        for path in ("lowlatency", "regular"):
            p = res["leads"][L][path]
            assert p["n_fields"] > 0, (L, path)
            assert p["csi"]["1.0"] == pytest.approx(1.0) and p["rmse"] == pytest.approx(0.0)
    assert res["leads"]["30"]["regular"]["mean_issue_age_min"] >= 30.0   # publish lag
    assert res["leads"]["30"]["lowlatency"]["mean_issue_age_min"] == 0.0


def test_newest_issue_at_uses_the_sorted_view_of_an_unsorted_store():
    ds = SimpleNamespace(_issue_epoch=np.array([100, 200, 400, 300, 500], dtype="int64"))  # 400/300 out of order
    ds._sorted_epoch = np.sort(ds._issue_epoch)
    ds._epoch_to_idx = {int(e): i for i, e in enumerate(ds._issue_epoch)}
    assert ll.newest_issue_at(ds, 350) == 3       # epoch 300 lives at raw index 3
    assert ll.newest_issue_at(ds, 450) == 2
    assert ll.newest_issue_at(ds, 50) is None
