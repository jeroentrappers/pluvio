"""Lagrangian input channels (TODO 2.3) — ZarrCorrectionDataset's opt-in
advected-observation planes, and the checkpoint channel recipe that lets
inference rebuild the same input.

The store here is not the shared ``synthetic_store`` fixture (random noise,
no coherent motion): these tests need a KNOWN displacement, so the radar
frames are crops of one smooth canvas whose window slides by a fixed
(dy, dx) each step. Frame ``i``'s content therefore sits ``i * (dy, dx)``
px further along than frame 0's, and the field advected to lead ``L`` has a
closed-form answer — the crop that frame ``i + L/step`` would be.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from model import zarr_dataset as zd
from model.infer_latest import dataset_for_checkpoint
from model.zarr_dataset import ZarrCorrectionDataset, advect_with_nan
from tests._store_spec import BOUNDS

G = 32                     # grid side — big enough for 4x4 NCC blocks of 8x8 px
N_ISSUES = 12
CADENCE_MIN = 30
LEADS_MIN = [0, 30, 60, 90]
HISTORY_STEPS = 3
TRUE_DY, TRUE_DX = 1, 2     # px per 30-min step (within the grid's search radius)
MAX_LEAD_STEPS = 3          # 90 min / 30 min
EPOCH0 = 1_700_000_000

# An NCC block on the south/east edge of the grid cannot slide its candidate
# window OUTWARD (motion.block_flow skips any offset that would leave the
# array), so with a southeastward true motion those blocks can only report
# offsets of the wrong sign. That is motion.py's own long-standing edge
# behaviour, not something these channels introduce — so the exactness checks
# below are made on the blocks that can actually search, and on the grid
# region their bilinear upsample covers: up to the centre of the last such
# block row/col (block rows of 8 px → centres 4, 12, 20, 28).
LAST_SEARCHING_CENTRE = 20
WARP_MARGIN = MAX_LEAD_STEPS * max(TRUE_DY, TRUE_DX) + 1  # warp clamps at the edges
CORE = (slice(WARP_MARGIN, LAST_SEARCHING_CENTRE),
        slice(WARP_MARGIN, LAST_SEARCHING_CENTRE))


def _smooth_canvas(rng, h: int, w: int) -> np.ndarray:
    """A smooth, non-periodic, everywhere-wet rain-like field: coarse white
    noise blown up 4x and box-blurred twice. Everywhere-wet matters — every
    NCC block must clear ``motion.MIN_WET_FRAC`` so the estimated flow is
    uniform, otherwise blocks with no signal are (correctly) flagged invalid
    and left at zero, and the bilinear upsample between block centres would
    dilute the very displacement under test."""
    coarse = rng.random(((h + 7) // 4, (w + 7) // 4))
    field = np.kron(coarse, np.ones((4, 4)))
    for _ in range(2):
        pad = np.pad(field, 1, mode="edge")
        field = sum(pad[dy:dy + field.shape[0], dx:dx + field.shape[1]]
                    for dy in range(3) for dx in range(3)) / 9.0
    return (0.5 + 7.5 * field[:h, :w]).astype("float32")


def _crop(canvas: np.ndarray, steps: float) -> np.ndarray:
    """The G x G frame whose content has moved ``steps * (TRUE_DY, TRUE_DX)``
    px from frame 0 — the window slides the opposite way to the content."""
    oy = round((N_ISSUES + MAX_LEAD_STEPS - steps) * TRUE_DY)
    ox = round((N_ISSUES + MAX_LEAD_STEPS - steps) * TRUE_DX)
    return canvas[oy:oy + G, ox:ox + G]


@pytest.fixture()
def motion_store(tmp_path):
    """Store whose radar frames advance by exactly (TRUE_DY, TRUE_DX) px per
    issue. Returns (path, canvas)."""
    rng = np.random.default_rng(7)
    canvas = _smooth_canvas(
        rng,
        G + (N_ISSUES + MAX_LEAD_STEPS + 1) * TRUE_DY,
        G + (N_ISSUES + MAX_LEAD_STEPS + 1) * TRUE_DX,
    )
    frames = np.stack([_crop(canvas, i) for i in range(N_ISSUES)])
    radar = np.repeat(frames[:, None], len(LEADS_MIN), axis=1).astype("float16")

    path = tmp_path / "motion.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.attrs.update({"grid_n": G, "bounds": list(BOUNDS), "store_version": 3,
                       "grid": "regular lat/lon, row 0 = north"})
    root.create_array("issue_time",
                      data=(EPOCH0 + np.arange(N_ISSUES) * CADENCE_MIN * 60).astype("int64"),
                      chunks="auto")
    root.create_array("leads_min", data=np.asarray(LEADS_MIN, dtype="int32"), chunks="auto")
    root.create_array("radar", data=radar, chunks=(4, len(LEADS_MIN), G, G))
    root.create_array("truth", data=frames.astype("float16"), chunks=(4, G, G))
    return path, canvas


def _dataset(path, lagrangian_channels: int) -> ZarrCorrectionDataset:
    return ZarrCorrectionDataset(
        path, leads_min=tuple(lead for lead in LEADS_MIN if lead),
        history_steps=HISTORY_STEPS, lagrangian_channels=lagrangian_channels,
    )


# ───────────────────────────────────────────────── channel count / layout


def test_channel_count_grows_by_the_option(motion_store):
    path, _ = motion_store
    base = _dataset(path, 0)
    assert base.n_channels == HISTORY_STEPS + 4 + len(base.aux_channels) + len(base.static_channels)
    assert _dataset(path, 1).n_channels == base.n_channels + 1
    assert _dataset(path, 2).n_channels == base.n_channels + 2


def test_option_off_leaves_build_input_bit_identical(motion_store):
    """Regression guard: the new planes are APPENDED, so every pre-existing
    channel keeps its index and its exact bytes whatever the option is."""
    path, _ = motion_store
    base, one, two = _dataset(path, 0), _dataset(path, 1), _dataset(path, 2)
    s = base.index[len(base.index) // 2]
    args = (s.issue_idx, s.lead_min, s.history_idx)
    x0, x1, x2 = base.build_input(*args), one.build_input(*args), two.build_input(*args)
    assert x0.shape[0] == base.n_channels
    np.testing.assert_array_equal(x0, x1[:base.n_channels])
    np.testing.assert_array_equal(x0, x2[:base.n_channels])
    np.testing.assert_array_equal(x1, x2[:one.n_channels])


def test_rejects_out_of_range_channel_count(motion_store):
    path, _ = motion_store
    with pytest.raises(ValueError, match="lagrangian_channels"):
        _dataset(path, 3)


# ──────────────────────────────────────────────────── the advected channel


@pytest.mark.parametrize("lead", [30, 60, 90])
def test_estimated_flow_matches_the_synthetic_motion(motion_store, lead):
    path, _ = motion_store
    ds = _dataset(path, 1)
    s = next(x for x in ds.index if x.lead_min == lead)
    vy, vx = ds.issue_block_flow(s.issue_idx, s.history_idx)
    assert np.abs(vy[:-1, :-1] - TRUE_DY).max() < 0.5
    assert np.abs(vx[:-1, :-1] - TRUE_DX).max() < 0.5


@pytest.mark.parametrize("lead", [30, 60, 90])
def test_advected_channel_is_the_frame_shifted_to_the_lead(motion_store, lead):
    """The advected plane at lead L must equal the newest analysis displaced
    by L/step * the true motion — i.e. the crop the *future* frame is."""
    path, canvas = motion_store
    ds = _dataset(path, 1)
    s = next(x for x in ds.index if x.lead_min == lead)
    chans = ds.build_input(s.issue_idx, s.lead_min, s.history_idx)
    got = chans[-1]

    steps = lead / CADENCE_MIN
    want = _crop(canvas, s.issue_idx + steps).astype("float32")
    # Integer motion + an exact flow estimate ⇒ warp copies source cells with
    # no interpolation, so the only slack is the store's float16 rounding.
    assert np.abs(got[CORE] - want[CORE]).max() < 0.02


def test_zero_flow_reduces_to_persistence(tmp_path):
    """A store with no motion at all: the estimator finds (0, 0) and the
    Lagrangian plane must be the latest analysis itself, not a blurred or
    drifted version of it."""
    rng = np.random.default_rng(3)
    frame = _smooth_canvas(rng, G, G)
    frames = np.repeat(frame[None], N_ISSUES, axis=0)
    radar = np.repeat(frames[:, None], len(LEADS_MIN), axis=1).astype("float16")

    path = tmp_path / "still.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.attrs.update({"grid_n": G, "bounds": list(BOUNDS), "store_version": 3,
                       "grid": "regular lat/lon, row 0 = north"})
    root.create_array("issue_time",
                      data=(EPOCH0 + np.arange(N_ISSUES) * CADENCE_MIN * 60).astype("int64"),
                      chunks="auto")
    root.create_array("leads_min", data=np.asarray(LEADS_MIN, dtype="int32"), chunks="auto")
    root.create_array("radar", data=radar, chunks=(4, len(LEADS_MIN), G, G))
    root.create_array("truth", data=frames.astype("float16"), chunks=(4, G, G))

    ds = _dataset(path, 1)
    s = ds.index[-1]
    vy, vx = ds.issue_block_flow(s.issue_idx, s.history_idx)
    assert not vy.any() and not vx.any()
    chans = ds.build_input(s.issue_idx, s.lead_min, s.history_idx)
    np.testing.assert_array_equal(chans[-1], chans[HISTORY_STEPS - 1])  # newest history plane


def test_flow_magnitude_plane_is_lead_independent_and_o1(motion_store):
    path, _ = motion_store
    ds = _dataset(path, 2)
    by_lead = {}
    for lead in (30, 60, 90):
        s = next(x for x in ds.index if x.lead_min == lead and x.issue_idx == 6)
        by_lead[lead] = ds.build_input(s.issue_idx, s.lead_min, s.history_idx)[-1]
    np.testing.assert_array_equal(by_lead[30], by_lead[90])
    expected = np.hypot(TRUE_DY, TRUE_DX) / ds.lagrangian_max_shift
    np.testing.assert_allclose(by_lead[30][CORE], expected, atol=0.02)
    assert float(by_lead[30].min()) >= 0.0 and float(by_lead[30].max()) <= 1.5


# ────────────────────────────────────────────────────── cost: flow caching


def test_flow_is_estimated_once_per_issue(motion_store, monkeypatch):
    """Cost guard: the flow depends on the (prev, issue) frame pair only, so
    all of an issue's leads must share one estimate."""
    path, _ = motion_store
    ds = _dataset(path, 1)
    calls = {"n": 0}
    real = zd.block_flow

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(zd, "block_flow", counting)
    samples = [s for s in ds.index if s.issue_idx == 6]
    assert len(samples) > 1
    for s in samples:
        ds.build_input(s.issue_idx, s.lead_min, s.history_idx)
    assert calls["n"] == 1


def test_single_step_history_gives_zero_flow(motion_store):
    """One history frame is no frame PAIR — the flow must be zero rather than
    a guess (Lagrangian persistence degenerates to persistence)."""
    path, _ = motion_store
    ds = ZarrCorrectionDataset(path, leads_min=(30,), history_steps=1,
                               lagrangian_channels=1)
    s = ds.index[0]
    vy, vx = ds.issue_block_flow(s.issue_idx, s.history_idx)
    assert not vy.any() and not vx.any()


# ─────────────────────────────────────────────────────────── NaN handling


def test_advect_with_nan_restores_the_source_nan_mask():
    field = np.zeros((16, 16), dtype="float32")
    field[4:8, 4:8] = 3.0
    field[:2, :] = np.nan                      # e.g. outside the radar domain
    dy = np.full((16, 16), 2.0, dtype="float32")
    dx = np.zeros((16, 16), dtype="float32")

    out = advect_with_nan(field, dy, dx)
    assert np.isnan(out[:4, :]).all()          # the NaN band moved down by 2
    assert np.isfinite(out[4:, :]).all()
    np.testing.assert_allclose(out[6:10, 4:8], 3.0, atol=1e-5)


def test_build_input_stays_finite_with_nan_in_the_radar_domain(motion_store):
    """The plane is NaN where nothing was advected in, but ``build_input``'s
    own nan_to_num is what the network sees — the same convention as every
    other channel."""
    path, _ = motion_store
    root = zarr.open_group(str(path), mode="r+")
    radar = root["radar"][:]
    radar[:, 0, :3, :] = np.nan
    root["radar"][:] = radar

    ds = _dataset(path, 2)
    s = ds.index[len(ds.index) // 2]
    chans = ds.build_input(s.issue_idx, s.lead_min, s.history_idx)
    assert np.isfinite(chans).all()
    assert float(chans[-2].max()) > 0.0  # not everything got zeroed away


# ──────────────────────────────────────────── checkpoint recipe round-trip


@pytest.mark.parametrize("lagrangian", [0, 1, 2])
def test_checkpoint_recipe_round_trip(motion_store, lagrangian):
    """A fake checkpoint (no torch.save needed): the recipe train.py records
    must rebuild a dataset whose build_input is identical, channel for
    channel, to the one that trained."""
    path, _ = motion_store
    trained = _dataset(path, lagrangian)
    ckpt = {"in_channels": trained.n_channels,
            "channel_recipe": trained.channel_recipe()}

    served = dataset_for_checkpoint(path, ckpt,
                                    tuple(lead for lead in LEADS_MIN if lead))
    assert served.n_channels == trained.n_channels
    assert served.lagrangian_channels == lagrangian
    assert served.history_steps == HISTORY_STEPS

    s = trained.index[len(trained.index) // 2]
    hist = served.history_for(s.issue_idx)
    assert hist == s.history_idx
    np.testing.assert_array_equal(
        trained.build_input(s.issue_idx, s.lead_min, s.history_idx),
        served.build_input(s.issue_idx, s.lead_min, hist),
    )


def test_recipeless_checkpoint_keeps_the_pre_2_3_input(motion_store):
    """A checkpoint trained before the recipe existed must resolve to zero
    Lagrangian channels and the dataset's own defaults."""
    path, _ = motion_store
    served = dataset_for_checkpoint(path, {}, (30,))
    assert served.lagrangian_channels == 0
    assert served.history_steps == zd.RADAR_HISTORY_STEPS


def test_recipe_channel_count_mismatch_raises(motion_store):
    path, _ = motion_store
    trained = _dataset(path, 1)
    ckpt = {"in_channels": trained.n_channels, "channel_recipe": trained.channel_recipe()}
    ckpt["channel_recipe"] = {**ckpt["channel_recipe"], "lagrangian_channels": 0}
    with pytest.raises(ValueError, match="input channels"):
        dataset_for_checkpoint(path, ckpt, (30,))


def test_cli_override_beats_the_recipe(motion_store):
    path, _ = motion_store
    trained = _dataset(path, 0)
    ckpt = {"channel_recipe": trained.channel_recipe()}
    ckpt["channel_recipe"] = {k: v for k, v in ckpt["channel_recipe"].items()
                              if k != "n_channels"}
    served = dataset_for_checkpoint(path, ckpt, (30,), lagrangian_channels=2)
    assert served.lagrangian_channels == 2
