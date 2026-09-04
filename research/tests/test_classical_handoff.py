"""The 2-6 h radar→NWP handoff must not make cells travel backwards.

Synthetic case modelled on the 2026-09-03 09:00Z measurement: a radar blob
moving east, NWP rain placed well to its west. A pointwise fade slides the
visible centre of mass west as the radar weight decays; the phase-corrected
blend keeps it monotone and still lands on the NWP's own placement.
"""

from __future__ import annotations

import numpy as np
import pytest

from model import classical

H = W = 64
LEADS = [0, 30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 600, 720, 900, 1080, 1200]


def _blob(cx: float, cy: float = 32.0, amp: float = 8.0, sigma: float = 4.0) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))).astype("float32")


def _centroid_col(field: np.ndarray, thr: float = 0.1) -> float:
    wet = field > thr
    if not wet.any():
        return float("nan")
    return float(np.nonzero(wet)[1].mean())


def _case(nwp_offset_px: float):
    v = 1.0                                   # px per 30-min step, eastward
    history = np.stack([_blob(20.0), _blob(20.0 + v)])   # two frames, dt 30 min
    nwp = np.stack([_blob(20.0 + v + v * (lead / 30.0) - nwp_offset_px) for lead in LEADS])
    return history, nwp


@pytest.mark.parametrize("phase_correct", [False, True])
def test_pointwise_fade_reverses_and_phase_correction_does_not(phase_correct):
    history, nwp = _case(nwp_offset_px=14.0)
    fc = classical.seamless_cube(history, LEADS, dt_min=30.0, aifs_rates=nwp,
                                 prefer_pysteps=False, phase_correct=phase_correct)
    cols = np.array([_centroid_col(f) for f in fc.rates])
    steps = np.diff(cols)
    blend = [i for i, l in enumerate(LEADS[1:], start=1)
             if classical.NOWCAST_END_MIN < l <= classical.BLEND_END_MIN]
    if not phase_correct:
        # the old fade: at least one westward step inside the blend window
        assert (steps[[i - 1 for i in blend]] < -0.5).any(), steps
        assert fc.phase_offset_px is None
    else:
        # corrected: never a westward step through the blend window (the
        # composition rides with the radar cells), and the relaxation that
        # follows in the outlook is a slow drift, never a jump
        through_blend = [i for i, l in enumerate(LEADS) if l <= classical.BLEND_END_MIN]
        assert (steps[: len(through_blend) - 1] >= -0.25).all(), steps
        assert (np.abs(steps) < 6.0).all(), steps
        assert fc.phase_offset_px is not None
        dy, dx = fc.phase_offset_px
        assert abs(dy) < 0.6 and dx == pytest.approx(14.0, abs=1.0)


def test_phase_correction_lands_on_nwp_after_the_relaxation():
    history, nwp = _case(nwp_offset_px=14.0)
    fc = classical.seamless_cube(history, LEADS, dt_min=30.0, aifs_rates=nwp,
                                 prefer_pysteps=False, phase_correct=True)
    relaxed = classical.BLEND_END_MIN + classical.PHASE_RELAX_MIN
    for i, lead in enumerate(LEADS):
        if lead >= relaxed:
            np.testing.assert_array_equal(fc.rates[i], np.clip(nwp[i], 0.0, None))
            assert fc.source[i] == "nwp"
        elif lead == classical.BLEND_END_MIN:
            # fully NWP-weighted but still co-located with the radar arm
            assert _centroid_col(fc.rates[i]) == pytest.approx(_centroid_col(nwp[i]) + 14.0, abs=1.5)


def test_colocated_nwp_is_bit_identical_to_the_plain_blend():
    history, nwp = _case(nwp_offset_px=0.0)
    a = classical.seamless_cube(history, LEADS, dt_min=30.0, aifs_rates=nwp,
                                prefer_pysteps=False, phase_correct=False)
    b = classical.seamless_cube(history, LEADS, dt_min=30.0, aifs_rates=nwp,
                                prefer_pysteps=False, phase_correct=True)
    assert b.phase_offset_px == (0.0, 0.0)
    np.testing.assert_array_equal(a.rates, b.rates)


def test_dry_or_wild_offsets_are_not_applied():
    _history, nwp = _case(nwp_offset_px=14.0)
    assert classical.nwp_phase_offset(np.zeros((H, W), "float32"), nwp[5]) == (0.0, 0.0)
    far = _blob(20.0 + 40.0)                      # > a quarter grid away
    assert classical.nwp_phase_offset(_blob(20.0), far) == (0.0, 0.0)


def test_phase_offset_sign_moves_nwp_onto_radar():
    radar = _blob(40.0)
    nwp = _blob(26.0)                              # rain 14 px west of the radar
    dy, dx = classical.nwp_phase_offset(radar, nwp)
    shifted = classical._advect_semilagrangian(nwp, dy, dx, 1)
    assert _centroid_col(shifted) == pytest.approx(_centroid_col(radar), abs=1.0)
