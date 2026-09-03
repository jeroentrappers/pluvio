"""model.zarr_dataset.ZarrCorrectionDataset against the synthetic store."""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from model.zarr_dataset import ZarrCorrectionDataset, issue_time_split

from tests._store_spec import BOUNDS, CADENCE_MIN, GRID_N, LEADS_MIN, N_ISSUES, NAN_ISSUE_IDX


def _broken_store(tmp_path, *, mutate):
    """A minimal valid store (radar/truth/issue_time/leads_min only), then
    `mutate(root)` adds/breaks one array before returning the path."""
    rng = np.random.default_rng(0)
    n, leads, g = N_ISSUES, len(LEADS_MIN), GRID_N
    issue_time = (1_700_000_000 + np.arange(n) * CADENCE_MIN * 60).astype("int64")

    path = tmp_path / "broken.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.attrs.update({"grid_n": g, "bounds": list(BOUNDS), "store_version": 3,
                       "grid": "regular lat/lon, row 0 = north"})
    root.create_array("issue_time", data=issue_time, chunks="auto")
    root.create_array("leads_min", data=np.asarray(LEADS_MIN, dtype="int32"), chunks="auto")
    root.create_array("radar", data=(rng.random((n, leads, g, g)) * 5.0).astype("float16"),
                      chunks=(16, leads, g, g))
    root.create_array("truth", data=(rng.random((n, g, g)) * 5.0).astype("float16"),
                      chunks=(16, g, g))
    mutate(root)
    return path


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


def test_discover_raises_on_mis_shaped_static(tmp_path):
    g = GRID_N

    def mutate(root):
        # named like a static channel but the wrong footprint — must not be
        # silently dropped.
        root.create_array("static_bad", data=np.zeros((g + 1, g), dtype="float16"))

    path = _broken_store(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="static_bad"):
        ZarrCorrectionDataset(path, leads_min=tuple(lead for lead in LEADS_MIN if lead))


def test_discover_raises_on_wrong_shaped_aux(tmp_path):
    n, g = N_ISSUES, GRID_N

    def mutate(root):
        # 3-D with the right issue count but the wrong grid footprint.
        root.create_array("msg_bad", data=np.zeros((n, g + 1, g), dtype="float16"))

    path = _broken_store(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="msg_bad"):
        ZarrCorrectionDataset(path, leads_min=tuple(lead for lead in LEADS_MIN if lead))


def test_discover_raises_on_unsupported_ndim(tmp_path):
    n, g = N_ISSUES, GRID_N

    def mutate(root):
        root.create_array("weird_4d", data=np.zeros((n, 2, g, g), dtype="float16"))

    path = _broken_store(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="weird_4d"):
        ZarrCorrectionDataset(path, leads_min=tuple(lead for lead in LEADS_MIN if lead))


def test_expected_channels_mismatch_raises(synthetic_store):
    with pytest.raises(ValueError, match="channels"):
        ZarrCorrectionDataset(
            synthetic_store,
            leads_min=tuple(lead for lead in LEADS_MIN if lead),
            expected_channels=999,
        )


def test_expected_channels_env_mismatch_raises(synthetic_store, monkeypatch):
    monkeypatch.setenv("PLUVIO_EXPECTED_CHANNELS", "999")
    with pytest.raises(ValueError, match="channels"):
        ZarrCorrectionDataset(synthetic_store, leads_min=tuple(lead for lead in LEADS_MIN if lead))


def test_open_raises_on_millisecond_issue_time(tmp_path):
    def mutate(root):
        root["issue_time"][:] = root["issue_time"][:] * 1000  # ms, not s

    path = _broken_store(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="epoch seconds"):
        ZarrCorrectionDataset(path, leads_min=tuple(lead for lead in LEADS_MIN if lead))


def test_issue_time_split_raises_on_millisecond_issue_time(tmp_path):
    def mutate(root):
        root["issue_time"][:] = root["issue_time"][:] * 1000

    path = _broken_store(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="epoch seconds"):
        issue_time_split(path, val_frac=0.2)


def test_open_warns_but_does_not_raise_on_zero_filled_issue_time_slot(tmp_path, caplog):
    # A zero-filled slot (e.g. build_zarr's resize-before-write crash window)
    # is not a units mixup — raising here would take down every 5-min
    # inference run over one bad slot.
    def mutate(root):
        t = root["issue_time"][:]
        t[0] = 0
        root["issue_time"][:] = t

    path = _broken_store(tmp_path, mutate=mutate)
    with caplog.at_level("WARNING"):
        ds = ZarrCorrectionDataset(path, leads_min=tuple(lead for lead in LEADS_MIN if lead))
    assert len(ds.index) > 0
    assert any("1 issue_time slot" in r.message for r in caplog.records)
