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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import zarr

from model import motion
from model import zarr_dataset as zd
from model.infer_latest import dataset_for_checkpoint
from model.zarr_dataset import (
    ZarrCorrectionDataset,
    advect_with_nan,
    free_blocks,
    repair_edge_flow,
)
from tests._store_spec import BOUNDS

G = 32                     # grid side — big enough for 4x4 NCC blocks of 8x8 px
N_ISSUES = 12
CADENCE_MIN = 30
LEADS_MIN = [0, 30, 60, 90]
HISTORY_STEPS = 3
TRUE_DY, TRUE_DX = 1, 2     # px per 30-min step (within the grid's search radius)
MAX_LEAD_STEPS = 3          # 90 min / 30 min
EPOCH0 = 1_700_000_000
STEP_SPAN = N_ISSUES - 1 + MAX_LEAD_STEPS   # furthest content offset any test needs


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


@dataclass
class MotionStore:
    """A store with a known uniform motion, plus the closed-form answer."""
    path: object
    dy: int
    dx: int
    crop: Callable[[float], np.ndarray]

    def unclamped(self, steps: float) -> tuple[slice, slice]:
        """The region where advecting by ``steps`` is a pure shift — outside
        it ``motion.warp`` clips the sampling coordinate at the grid edge, so
        there is no "true" answer to compare against. Note this trims the
        UPSTREAM border only: the downstream edge bands, where the raw
        block-flow estimator reports zero motion, stay in the comparison."""
        def axis(shift: float) -> slice:
            s = round(shift)
            return slice(max(s, 0), G + min(s, 0))
        return axis(steps * self.dy), axis(steps * self.dx)

    def downstream_band(self, steps: float) -> tuple[slice, slice]:
        """The last block row/col in each direction of travel — the band the
        unrepaired estimator leaves stationary."""
        band = G // motion.BLOCKS
        rows, cols = self.unclamped(steps)
        return ((slice(G - band, G) if self.dy > 0 else
                 slice(0, band) if self.dy < 0 else rows),
                (slice(G - band, G) if self.dx > 0 else
                 slice(0, band) if self.dx < 0 else cols))


def _make_store(tmp_path, dy: int, dx: int, name: str = "motion.zarr") -> MotionStore:
    """Radar frames that advance by exactly (dy, dx) px per issue."""
    rng = np.random.default_rng(7)
    canvas = _smooth_canvas(rng, G + STEP_SPAN * abs(dy), G + STEP_SPAN * abs(dx))
    # Content moves by +steps*(dy, dx), so the crop window slides the other
    # way; anchor it so every window (steps 0..STEP_SPAN) is inside the canvas
    # for either sign of the motion.
    oy0, ox0 = max(0, STEP_SPAN * dy), max(0, STEP_SPAN * dx)

    def crop(steps: float) -> np.ndarray:
        oy, ox = round(oy0 - steps * dy), round(ox0 - steps * dx)
        return canvas[oy:oy + G, ox:ox + G]

    frames = np.stack([crop(i) for i in range(N_ISSUES)])
    radar = np.repeat(frames[:, None], len(LEADS_MIN), axis=1).astype("float16")

    path = tmp_path / name
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.attrs.update({"grid_n": G, "bounds": list(BOUNDS), "store_version": 3,
                       "grid": "regular lat/lon, row 0 = north"})
    root.create_array("issue_time",
                      data=(EPOCH0 + np.arange(N_ISSUES) * CADENCE_MIN * 60).astype("int64"),
                      chunks="auto")
    root.create_array("leads_min", data=np.asarray(LEADS_MIN, dtype="int32"), chunks="auto")
    root.create_array("radar", data=radar, chunks=(4, len(LEADS_MIN), G, G))
    root.create_array("truth", data=frames.astype("float16"), chunks=(4, G, G))
    return MotionStore(path=path, dy=dy, dx=dx, crop=crop)


@pytest.fixture()
def motion_store(tmp_path) -> MotionStore:
    return _make_store(tmp_path, TRUE_DY, TRUE_DX)


def _dataset(path, lagrangian_channels: int, **kw) -> ZarrCorrectionDataset:
    return ZarrCorrectionDataset(
        path, leads_min=tuple(lead for lead in LEADS_MIN if lead),
        history_steps=HISTORY_STEPS, lagrangian_channels=lagrangian_channels, **kw,
    )


def _sample(ds, lead: int, issue_idx: int | None = None):
    return next(s for s in ds.index
                if s.lead_min == lead and (issue_idx is None or s.issue_idx == issue_idx))


# ───────────────────────────────────────────────── channel count / layout


def test_channel_count_grows_by_the_option(motion_store):
    base = _dataset(motion_store.path, 0)
    assert base.n_channels == HISTORY_STEPS + 4 + len(base.aux_channels) + len(base.static_channels)
    assert _dataset(motion_store.path, 1).n_channels == base.n_channels + 1
    assert _dataset(motion_store.path, 2).n_channels == base.n_channels + 2


def test_option_off_leaves_build_input_bit_identical(motion_store):
    """Regression guard: the new planes are APPENDED, so every pre-existing
    channel keeps its index and its exact bytes whatever the option is."""
    path = motion_store.path
    base, one, two = _dataset(path, 0), _dataset(path, 1), _dataset(path, 2)
    s = base.index[len(base.index) // 2]
    args = (s.issue_idx, s.lead_min, s.history_idx)
    x0, x1, x2 = base.build_input(*args), one.build_input(*args), two.build_input(*args)
    assert x0.shape[0] == base.n_channels
    np.testing.assert_array_equal(x0, x1[:base.n_channels])
    np.testing.assert_array_equal(x0, x2[:base.n_channels])
    np.testing.assert_array_equal(x1, x2[:one.n_channels])


def test_channel_names_cover_every_plane_in_order(motion_store):
    for lag in (0, 1, 2):
        ds = _dataset(motion_store.path, lag)
        names = ds.channel_names()
        assert len(names) == ds.n_channels
        assert names[ds.history_steps] == "nowcast_at_lead"
        assert names[ds.history_steps:ds.history_steps + 4] == [
            "nowcast_at_lead", "lead_over_120", "tod_sin", "tod_cos"]
        assert names[base_len(ds):] == ["lagrangian_rate", "lagrangian_flow_mag"][:lag]


def base_len(ds) -> int:
    return ds.n_channels - ds.lagrangian_channels


def test_rejects_out_of_range_channel_count(motion_store):
    with pytest.raises(ValueError, match="lagrangian_channels"):
        _dataset(motion_store.path, 3)


# ──────────────────────────────────────────────────── the advected channel


@pytest.mark.parametrize("lead", [30, 60, 90])
def test_estimated_flow_matches_the_synthetic_motion(motion_store, lead):
    ds = _dataset(motion_store.path, 1)
    s = _sample(ds, lead)
    vy, vx = ds.issue_block_flow(s.issue_idx, s.history_idx)
    assert np.abs(vy - TRUE_DY).max() < 0.5
    assert np.abs(vx - TRUE_DX).max() < 0.5


@pytest.mark.parametrize("lead", [30, 60, 90])
def test_advected_channel_is_the_frame_shifted_to_the_lead(motion_store, lead):
    """The advected plane at lead L must equal the newest analysis displaced
    by L/step * the true motion — i.e. the crop the *future* frame is."""
    ds = _dataset(motion_store.path, 1)
    s = _sample(ds, lead)
    got = ds.build_input(s.issue_idx, s.lead_min, s.history_idx)[-1]

    steps = lead / CADENCE_MIN
    want = motion_store.crop(s.issue_idx + steps).astype("float32")
    # Integer motion + an exact flow estimate ⇒ warp copies source cells with
    # no interpolation, so the only slack is the store's float16 rounding.
    region = motion_store.unclamped(steps)
    assert np.abs(got[region] - want[region]).max() < 0.02


# ─────────────────────────────────────── the downstream-edge repair (blocker)


@pytest.mark.parametrize("dy,dx", [(3, 3), (-3, 3), (0, 4)])
def test_every_block_reports_the_motion_after_the_edge_repair(tmp_path, dy, dx):
    """``motion.block_flow`` cannot slide an edge block's window outward, so
    the blocks on the two DOWNSTREAM edges cannot see the displacement and
    report ~zero with valid=True. ``repair_edge_flow`` fills them from the
    blocks that could look, so the whole field carries the motion."""
    st = _make_store(tmp_path, dy, dx)
    ds = _dataset(st.path, 1)
    s = _sample(ds, 90)
    vy, vx = ds.issue_block_flow(s.issue_idx, s.history_idx)
    assert np.abs(vy - dy).max() < 0.5, vy
    assert np.abs(vx - dx).max() < 0.5, vx


@pytest.mark.parametrize("dy,dx", [(3, 3), (-3, 3), (0, 4)])
def test_no_stationary_band_in_the_advected_plane_at_the_longest_lead(tmp_path, dy, dx):
    """The artefact this repair exists for: without it the advected plane is
    correct in the interior and *unmoved* along the two downstream edges,
    with a linear decay between. Checked over the whole grid except the
    upstream border warp legitimately clamps — the downstream bands very much
    included, and then again on their own."""
    st = _make_store(tmp_path, dy, dx)
    ds = _dataset(st.path, 1)
    s = _sample(ds, 90)
    got = ds.build_input(s.issue_idx, s.lead_min, s.history_idx)[-1]
    want = st.crop(s.issue_idx + MAX_LEAD_STEPS).astype("float32")

    region = st.unclamped(MAX_LEAD_STEPS)
    assert np.abs(got[region] - want[region]).max() < 0.02
    band = st.downstream_band(MAX_LEAD_STEPS)
    assert np.abs(got[band] - want[band]).max() < 0.02
    # ...and the plane really did move there (a coincidence-proof check: the
    # unadvected latest analysis does NOT match in that band).
    latest = st.crop(s.issue_idx).astype("float32")
    assert np.abs(latest[band] - want[band]).max() > 0.1


def test_raw_block_flow_still_needs_the_repair(tmp_path):
    """Documents WHY repair_edge_flow exists, measured on the same store: the
    unrepaired estimator gets the downstream row/col wrong. If this ever
    fails because motion.block_flow learned to search outward, delete
    repair_edge_flow instead of loosening this."""
    st = _make_store(tmp_path, 3, 3)
    ds = _dataset(st.path, 1)
    s = _sample(ds, 90)
    radar = zarr.open_group(str(st.path), mode="r")["radar"]
    prev = np.asarray(radar[s.history_idx[-2], 0], dtype="float32")
    curr = np.asarray(radar[s.issue_idx, 0], dtype="float32")
    vy, vx, valid = motion.block_flow(prev, curr, max_shift=ds.lagrangian_max_shift)

    assert valid.all()                       # every block had rain: not a dry-block story
    assert abs(vy[1, 1] - 3) < 0.5           # a free interior block is fine
    assert vy[-1, :].max() < 1.0             # the downstream row reports ~no motion
    assert vx[:, -1].max() < 1.0
    fixed_y, fixed_x = repair_edge_flow(vy, vx, valid, ds.grid_hw, ds.lagrangian_max_shift)
    assert np.abs(fixed_y - 3).max() < 0.5
    assert np.abs(fixed_x - 3).max() < 0.5


def test_free_blocks_marks_only_the_blocks_that_can_search_both_ways():
    free = free_blocks((32, 32), 4, 4)       # 8-px blocks, ±4 px search
    assert free.shape == (4, 4)
    assert not free[0, :].any() and not free[-1, :].any()
    assert not free[:, 0].any() and not free[:, -1].any()
    assert free[1:-1, 1:-1].all()
    # a search radius that fits nowhere leaves nothing free
    assert not free_blocks((32, 32), 4, 12).any()


def test_repair_edge_flow_also_fills_dry_blocks_and_survives_an_empty_field():
    vy = np.zeros((4, 4), "float32")
    vx = np.zeros((4, 4), "float32")
    vy[1:-1, 1:-1] = 2.0
    vx[1:-1, 1:-1] = -1.0
    valid = np.ones((4, 4), bool)
    valid[2, 2] = False                      # a dry interior block
    vy[2, 2] = vx[2, 2] = 0.0
    fy, fx = repair_edge_flow(vy, vx, valid, (32, 32), 4)
    np.testing.assert_allclose(fy, 2.0)      # median of the free+valid blocks
    np.testing.assert_allclose(fx, -1.0)

    # nothing measured anywhere: hand the estimate back untouched
    zeros = np.zeros((4, 4), "float32")
    fy, fx = repair_edge_flow(zeros, zeros, np.zeros((4, 4), bool), (32, 32), 4)
    assert not fy.any() and not fx.any()


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
    ds = _dataset(motion_store.path, 2)
    by_lead = {lead: ds.build_input(*_lead_args(ds, lead))[-1] for lead in (30, 60, 90)}
    np.testing.assert_array_equal(by_lead[30], by_lead[90])
    expected = np.hypot(TRUE_DY, TRUE_DX) / ds.lagrangian_max_shift
    np.testing.assert_allclose(by_lead[30], expected, atol=0.02)
    assert float(by_lead[30].min()) >= 0.0 and float(by_lead[30].max()) <= 1.5


def _lead_args(ds, lead: int):
    s = _sample(ds, lead, issue_idx=6)
    return s.issue_idx, s.lead_min, s.history_idx


# ────────────────────────────────────────────────────── cost: flow caching


def test_flow_is_estimated_once_per_issue(motion_store, monkeypatch):
    """Cost guard: the flow depends on the (prev, issue) frame pair only, so
    all of an issue's leads must share one estimate."""
    ds = _dataset(motion_store.path, 1)
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


def test_flow_cache_is_keyed_on_the_frame_pair(motion_store):
    """The same issue reached with a different predecessor (a coarser history
    step) is a different frame pair and must not reuse the cached flow."""
    ds = _dataset(motion_store.path, 1)
    s = _sample(ds, 30, issue_idx=6)
    fine = ds.issue_block_flow(s.issue_idx, s.history_idx)
    coarse_hist = (s.history_idx[0], s.issue_idx)          # a 2-step-back predecessor
    coarse = ds.issue_block_flow(s.issue_idx, coarse_hist)
    assert set(ds._flow_cache) == {(s.issue_idx, s.history_idx[-2]),
                                   (s.issue_idx, coarse_hist[-2])}
    assert abs(float(coarse[0].mean()) - 2 * float(fine[0].mean())) < 0.5


def test_single_step_history_gives_zero_flow(motion_store):
    """One history frame is no frame PAIR — the flow must be zero rather than
    a guess (Lagrangian persistence degenerates to persistence)."""
    ds = ZarrCorrectionDataset(motion_store.path, leads_min=(30,), history_steps=1,
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
    root = zarr.open_group(str(motion_store.path), mode="r+")
    radar = root["radar"][:]
    radar[:, 0, :3, :] = np.nan
    root["radar"][:] = radar

    ds = _dataset(motion_store.path, 2)
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
    path = motion_store.path
    trained = _dataset(path, lagrangian)
    ckpt = {"in_channels": trained.n_channels,
            "channel_recipe": trained.channel_recipe()}

    served = dataset_for_checkpoint(path, ckpt,
                                    tuple(lead for lead in LEADS_MIN if lead))
    assert served.n_channels == trained.n_channels
    assert served.lagrangian_channels == lagrangian
    assert served.history_steps == HISTORY_STEPS
    assert served.history_tolerance_s == trained.history_tolerance_s

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
    served = dataset_for_checkpoint(motion_store.path, {}, (30,))
    assert served.lagrangian_channels == 0
    assert served.history_steps == zd.RADAR_HISTORY_STEPS


def test_recipe_channel_count_mismatch_raises(motion_store):
    trained = _dataset(motion_store.path, 1)
    ckpt = {"in_channels": trained.n_channels, "channel_recipe": trained.channel_recipe()}
    ckpt["channel_recipe"] = {**ckpt["channel_recipe"], "lagrangian_channels": 0}
    with pytest.raises(ValueError, match="input channels"):
        dataset_for_checkpoint(motion_store.path, ckpt, (30,))


def test_renamed_static_channel_raises_even_at_the_same_count(motion_store):
    """A static renamed between training and serving keeps the channel count
    intact, so only a name comparison catches it."""
    trained = _dataset(motion_store.path, 1)
    recipe = {**trained.channel_recipe(), "static_channels": ["static_elevation_m"]}
    ckpt = {"in_channels": trained.n_channels + 1, "channel_recipe": recipe}
    with pytest.raises(ValueError, match="static channels"):
        dataset_for_checkpoint(motion_store.path, ckpt, (30,))


def test_cli_override_beats_the_recipe(motion_store):
    trained = _dataset(motion_store.path, 0)
    recipe = {k: v for k, v in trained.channel_recipe().items() if k != "n_channels"}
    served = dataset_for_checkpoint(motion_store.path, {"channel_recipe": recipe}, (30,),
                                    lagrangian_channels=2)
    assert served.lagrangian_channels == 2
