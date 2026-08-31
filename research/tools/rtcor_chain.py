"""Replicate the KNMI RTCOR single-radar chain on our own volume sources.

Why: RTCOR beats our composite in 15 of 16 configurations and beats OPERA everywhere,
so it is the bar. Its chain is published (Overeem et al. 2025, ESSD 17:4715) and this
follows it step by step rather than deriving alternatives. Where the paper gives the
exact form it is used verbatim; where it defers to another paper the simplest faithful
version is used and flagged.

Per-voxel processing on the polar volume, then quality-weighted merging to the grid:

  1. Non-meteorological echo removal — the SAME fuzzy-logic classifier RTCOR uses:
     the paper names wradlib's `classify_echo_fuzzy` explicitly, with weights
     texture(ZDR) 0.20, RhoHV 0.15, texture(RhoHV) 0.25, depolarization ratio 0.20,
     clutter phase alignment 0.20 and a 0.6 threshold (Sect. 3.2). All five variables
     exist in the KNMI volumes (CPA included). Voxels below threshold get quality 0;
     survivors carry Q_F = S(F, 0.8, 0.95) (Eq. A2).
  2. Attenuation correction from K_dp:  A_h = 0.081 * K_dp  (dB/km, Sect. 3.3), two-way
     PIA integrated outward along the ray. Quality reduction Q_A = exp(-ln2 (A/A0)^2),
     A0 = 3 dB (Eq. A4). Applied only where the voxel is below the freezing level; a
     fixed 2.5 km stand-in is used until an NWP freezing level is wired in.
  3. Height quality Q_H (Eq. A5): rises through the lowest 500 m, ~0.95 from 0.5-1 km,
     falls to ~0.05 at 4 km.  Range quality Q_R (Eqs. A6-A8): linear to zero at 500 km,
     tapered over the last 50 km of the scan.
  4. Q_T = Q_C Q_F Q_A Q_V Q_U Q_H Q_R (Eq. A9). Q_C (Doppler filter correction) is not
     available in our files and is set to 1; Q_V/Q_U (VPR) are 1 until VPR is added.
  5. Quality-weighted scan merging on the grid:  Z_Q = sum(Q_T Z)/sum(Q_T) in LINEAR Z,
     combined quality Q_r = 1 - prod(1 - Q_T)  (Eqs. 1-2).
  6. Gabella residual-clutter filter on the merged Cartesian map (Sect. 3.6): a pixel is
     clutter if fewer than 6 of the 5x5 neighbourhood are within 6 dB below it, or if a
     contiguous echo area above 0 dBZ has area/circumference < 1.3.
  7. Marshall-Palmer Z = 200 R^1.6 (Eq. 3).

Compositing across radars reuses step 5 with the per-radar Q_r as weight.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np

LOG = logging.getLogger("pluvio.rtcor_chain")

# --- Appendix A constants -------------------------------------------------------------
C0_DB = 3.0          # halving-quality correction, Doppler filter (A1)
A0_DB = 3.0          # halving-quality correction, attenuation (A4)
F5, F95 = 0.8, 0.95  # fuzzy score at which quality is 0.05 / 0.95 (A2)
H_L, H_M, H_H = 0.5, 1.0, 4.0   # km, height quality breakpoints (A5)
R_M = 500.0          # km, range at which quality reaches 0 (A6)
R_E = 50.0           # km, taper at end of scan (A7)
# --- Sect. 3.3 -------------------------------------------------------------------------
KDP_ATTEN_COEF = 0.081     # dB/km per deg/km, horizontal
FREEZING_LEVEL_M = 2500.0  # stand-in until an NWP freezing level is wired in
# --- Sect. 3.2: fuzzy-logic clutter classification --------------------------------------
FUZZY_THRESHOLD = 0.6          # below this the voxel is clutter (Dutch radars)
FUZZY_WEIGHTS = {              # paper weights; unused wradlib variables weighted 0
    "zdr": 0.20,   # texture of ZDR
    "rho": 0.25,   # texture of RhoHV
    "rho2": 0.15,  # RhoHV itself
    "dr": 0.20,    # depolarization ratio (Ryzhkov et al. 2017)
    "cpa": 0.20,   # clutter phase alignment (Hubbert et al. 2009)
    "phi": 0.0, "dop": 0.0, "map": 0.0,
}
# Overeem et al. 2020 (JTECH 37:1643) use wradlib's default trapezoids EXCEPT RhoHV:
# 0.80-0.85 instead of wradlib's 0.95-0.98, "because values below 0.8 are generally
# associated with nonmeteorological scatterers". With the wradlib default, ordinary rain
# with RhoHV 0.90-0.95 collects clutter membership and the flag rate is far too high.
FUZZY_TRPZ = {"rho2": [-9999.0, -9999.0, 0.80, 0.85]}
RHOHV_MIN = 0.80               # fallback when the dual-pol set is incomplete
# Bins below this hold no meaningful precipitation signal (Marshall-Palmer at -25 dBZ is
# ~0.0004 mm/h). The clutter classifier must only judge bins WITH echo: DWD's
# polarimetric moments come unfiltered and carry noise in echo-free bins, and running
# the classifier there "removed" 99.5% of a dry scan — deleting the measurement that it
# was dry. Dry bins keep their value and full Q_F.
ECHO_MIN_DBZ = -25.0
KE = 4.0 / 3.0 * 6371000.0


def fuzzy_meteo_score(sw):
    """Probability that each voxel is meteorological, per RTCOR's own classifier.

    Uses wradlib's classify_echo_fuzzy — the implementation the paper cites — with the
    paper's weights. Returns None when the sweep lacks the dual-pol moments (then the
    RhoHV fallback applies instead).
    """
    import wradlib.classify as classify
    import wradlib.dp as dp
    import wradlib.util as util

    if sw.get("zdr") is None or sw.get("rhohv") is None:
        return None
    # ⚠️ NaN must be PRESERVED here. Filling the void around echo with zeros manufactures
    # sharp gradients at every echo edge, the textures explode, and the classifier flags
    # a third of genuine rain as clutter (measured: 36% of echo on a stratiform morning).
    # With NaN kept, texture is undefined off-echo and the membership stays neutral.
    zdr = sw["zdr"]
    rho = np.clip(sw["rhohv"], 0.0, 1.0)
    dat = {
        "zdr": util.texture(zdr),
        "rho": util.texture(rho),
        "rho2": rho,
        "dr": dp.depolarization(np.nan_to_num(zdr, nan=0.0),
                                np.clip(np.nan_to_num(rho, nan=1.0), 0.001, 0.9999)),
        "cpa": sw["cpa"] if sw.get("cpa") is not None else np.zeros_like(zdr),
        # mandatory keys the classifier requires but the paper weights at 0
        "phi": np.zeros_like(zdr), "dop": np.zeros_like(zdr), "map": np.zeros_like(zdr),
    }
    weights = dict(FUZZY_WEIGHTS)
    if sw.get("cpa") is None:
        weights["cpa"] = 0.0
    prob, _ = classify.classify_echo_fuzzy(dat, weights=weights, trpz=FUZZY_TRPZ)
    prob = np.asarray(prob)
    # Where the score is undefined (moments missing) there is no evidence either way:
    # keep the voxel and let RhoHV alone speak through Q_F.
    return np.where(np.isfinite(prob), prob,
                    np.where(np.isfinite(rho), rho, 1.0))


def sigmoid(x, x5, x95):
    """Eq. A3: S(x) = 0.05 at x5 and 0.95 at x95."""
    k = 2.0 * np.log(0.95 / 0.05) / (x5 - x95)
    return 1.0 / (1.0 + np.exp(k * (x - (x5 + x95) / 2.0)))


def q_gauss(correction_db, half_db):
    """Eqs. A1/A4: quality halves at `half_db` of applied correction."""
    return np.exp(-np.log(2.0) * (np.abs(correction_db) / half_db) ** 2)


def q_height(h_km):
    """Eq. A5 — with the formula taken from the paper's stated BEHAVIOUR, not its typography.

    The PDF extraction reads "QH = S(h,0,hl) − 0.05/0.95 · S(h,hh,hm)", but that
    expression cannot satisfy the paper's own endpoints: since S(h, 4.0, 1.0) = 0.05 at
    h = 4 by definition of S, the subtractive form gives QH(4 km) ≈ 1 − 0.0526·0.05 ≈
    1.0 where the text demands "decreases to approximately 0.05 at hh = 4.0 km".
    Measured consequence of implementing the typography: behel carried Q = 0.95 at
    200 km range (beam ~3.4 km), so distant overshooting radars outvoted near ones with
    legitimate dry-aloft readings and composite POD collapsed to 0.35.

    The product of the two sigmoids reproduces every stated property: rising through
    the lowest 500 m, ~0.95 plateau from 0.5-1 km, 0.05 at 4 km, →0 above.
    """
    return sigmoid(h_km, 0.0, H_L) * sigmoid(h_km, H_H, H_M)


def q_range(r_km, r_max_km):
    """Eqs. A6-A8."""
    q_m = np.where(r_km < R_M, 1.0 - r_km / R_M, 0.0)
    tail = r_km > (r_max_km - R_E)
    q_e = np.where(tail, 1.0 - ((r_km - r_max_km) / R_E + 1.0) ** 2, 1.0)
    return np.clip(q_m * np.clip(q_e, 0.0, 1.0), 0.0, 1.0)


def beam_height_m(rng_m, elangle_deg, alt_m):
    """Beam-centre height above sea level along the ray, 4/3-earth."""
    el = np.radians(elangle_deg)
    return alt_m + np.sqrt(rng_m ** 2 + KE ** 2 + 2 * rng_m * KE * np.sin(el)) - KE


def attenuation_correct(dbz, kdp, rng_m, height_m):
    """Sect. 3.3: A_h = 0.081 K_dp, two-way PIA accumulated outward.

    K_dp below zero is noise, not negative attenuation, and is clipped. Voxels above the
    freezing level do not contribute to the path integral (ice attenuates far less and
    K_dp there is unreliable), though the PIA accumulated below them still applies.
    Returns (corrected dbz, PIA in dB).
    """
    if kdp is None:
        return dbz, np.zeros_like(dbz)
    dr_km = np.gradient(rng_m) / 1000.0
    k = np.nan_to_num(np.clip(kdp, 0.0, None), nan=0.0)
    k = np.where(height_m < FREEZING_LEVEL_M, k, 0.0)
    pia = 2.0 * np.cumsum(KDP_ATTEN_COEF * k * dr_km[None, :], axis=1)
    return dbz + pia, pia


def process_sweep(sw):
    """One sweep -> (dbz_corrected, Q_T) on its own polar grid."""
    dbz = sw["dbz"].copy()
    rng = sw["rng"]
    alt = sw["site"][2] if len(sw["site"]) > 2 else 0.0
    h = beam_height_m(rng, sw["elangle"], alt)[None, :].repeat(dbz.shape[0], 0)

    # 1. non-meteorological echo: RTCOR's fuzzy-logic classifier, RhoHV fallback.
    # Only bins with echo are judged — see ECHO_MIN_DBZ above.
    echo = np.isfinite(dbz) & (dbz > ECHO_MIN_DBZ)
    score = fuzzy_meteo_score(sw)
    if score is not None:
        bad = echo & (score < FUZZY_THRESHOLD)
        dbz[bad] = np.nan
        q_f = np.where(echo, sigmoid(score, F5, F95), 1.0)   # Eq. A2 on echo only
    else:
        q_f = np.ones_like(dbz)
        if sw.get("rhohv") is not None:
            rho = sw["rhohv"]
            bad = echo & np.isfinite(rho) & (rho < RHOHV_MIN)
            dbz[bad] = np.nan
            q_f = np.where(echo & np.isfinite(rho), sigmoid(rho, F5, F95), 1.0)

    # 2. attenuation
    dbz, pia = attenuation_correct(dbz, sw.get("kdp"), rng, h)
    q_a = q_gauss(pia, A0_DB)

    # 3. geometry
    r_km = rng[None, :] / 1000.0
    q_h = q_height((h - alt) / 1000.0)          # height above the radar's ground
    q_r = q_range(r_km, rng.max() / 1000.0)

    q_t = q_f * q_a * q_h * q_r
    q_t = np.where(np.isfinite(dbz), q_t, 0.0)
    return dbz, q_t


# --- Sect. 3.4: vertical profile of reflectivity ----------------------------------------
VPR_REF_KM = 0.8       # extrapolate to this height ("ground" for a beam-broadened radar)
VPR_MAX_DB = 12.0      # cap on the applied correction
VPR_CONV_DBZ = 40.0    # voxels above this are convective: no VPR correction (Sect. 3.4)


def estimate_vpr(sweeps, r_min_km=20.0, r_max_km=100.0, dz_km=0.2, top_km=8.0):
    """Idealized stratiform VPR fitted to the volume, Hazenberg-style.

    v1 used the raw apparent profile, which is nearly useless: beam broadening smears
    the bright band flat (measured: 13-14 dBZ from 0.5-3 km on a stratiform morning, so
    corrections were ~0 and the chain kept under-reading at range — exactly where RTCOR
    keeps its POD edge). v2 fits the paper's idealized shape instead:

        P(h) = 0                       below the melting layer      (rain, reference)
        P(h) = triangular bump         inside it                    (bright band)
        P(h) = -s * (h - ml_top)       above it                     (snow fall-off)

    The melting layer comes from the polarimetric dip (Boodoo et al. 2010): rain and
    dry snow have RhoHV ~0.99, melting hydrometeors drop it. Bright-band amplitude and
    the snow slope s are fitted from the apparent profile, so the idealization is
    anchored to this volume's own precipitation. The evaluated correction is
    beam-weighted (Gaussian, 1 deg beam), which is what lets it stay meaningful at
    ranges where the beam is a kilometre thick.

    Returns an opaque dict for vpr_correction_db, or None (no stratiform echo).
    """
    edges = np.arange(0.0, top_km + dz_km, dz_km)
    mids = (edges[:-1] + edges[1:]) / 2.0
    zsum = np.zeros(len(mids)); zcnt = np.zeros(len(mids))
    rsum = np.zeros(len(mids)); rcnt = np.zeros(len(mids))
    alt = sweeps[0]["site"][2] if len(sweeps[0]["site"]) > 2 else 0.0
    for sw in sweeps:
        rng = sw["rng"]
        sel = (rng >= r_min_km * 1000.0) & (rng <= r_max_km * 1000.0)
        if not sel.any():
            continue
        h_km = (beam_height_m(rng[sel], sw["elangle"], alt) - alt) / 1000.0
        idx = np.clip(np.digitize(h_km, edges) - 1, 0, len(mids) - 1)
        dbz = sw["dbz"][:, sel]
        strat = np.isfinite(dbz) & (dbz > 5.0) & (dbz < VPR_CONV_DBZ)
        rho = sw.get("rhohv")
        rho = rho[:, sel] if rho is not None else None
        for j in np.unique(idx):
            col = idx == j
            v = dbz[:, col][strat[:, col]]
            if v.size:
                zsum[j] += np.sum(10.0 ** (v / 10.0)); zcnt[j] += v.size
            if rho is not None:
                rv = rho[:, col][strat[:, col]]
                rv = rv[np.isfinite(rv)]
                if rv.size:
                    rsum[j] += rv.sum(); rcnt[j] += rv.size
    ok = zcnt > 100
    if ok.sum() < 6:
        return None
    app_db = np.full(len(mids), np.nan)
    app_db[ok] = 10.0 * np.log10(zsum[ok] / zcnt[ok])
    rho_prof = np.where(rcnt > 100, rsum / np.maximum(rcnt, 1), np.nan)

    # Melting layer: the RhoHV dip. Rain and dry snow sit ~0.99; melting drops it.
    ml_peak = None
    cand = np.where(np.isfinite(rho_prof) & (mids > 0.4) & (mids < 5.0))[0]
    if cand.size:
        dip = cand[np.argmin(rho_prof[cand])]
        if rho_prof[dip] < 0.97:
            ml_peak = mids[dip]
    if ml_peak is None:
        ml_peak = FREEZING_LEVEL_M / 1000.0        # no polarimetric dip visible
    ml_bot, ml_top = ml_peak - 0.35, ml_peak + 0.35

    # Anchor the idealization to the volume: rain level, bright-band amplitude, snow slope.
    rain_sel = ok & (mids >= 0.4) & (mids < ml_bot)
    snow_sel = ok & (mids > ml_top + 0.2) & (mids < ml_top + 3.5)
    if not rain_sel.any() or snow_sel.sum() < 3:
        return None
    rain_db = np.nanmean(app_db[rain_sel])
    bb_sel = ok & (mids >= ml_bot) & (mids <= ml_top)
    bb_amp = float(np.clip((np.nanmax(app_db[bb_sel]) - rain_db) if bb_sel.any() else 0.0, 0.0, 7.0))
    slope, icpt = np.polyfit(mids[snow_sel], app_db[snow_sel], 1)   # dB per km, <0 expected
    slope = float(np.clip(slope, -10.0, -1.0))
    snow_at_top = float(np.clip(icpt + slope * ml_top - rain_db, -3.0, 3.0))

    # Idealized relative profile on a fine grid, then beam-weighted per (height, sigma).
    hgrid = np.arange(0.0, top_km, 0.05)
    prof = np.zeros_like(hgrid)
    in_ml = (hgrid >= ml_bot) & (hgrid <= ml_top)
    prof[in_ml] = bb_amp * (1.0 - np.abs(hgrid[in_ml] - ml_peak) / max(ml_peak - ml_bot, 1e-6))
    above = hgrid > ml_top
    prof[above] = snow_at_top + slope * (hgrid[above] - ml_top)
    fit_res = app_db[snow_sel] - (icpt + slope * mids[snow_sel])
    unc_db = float(np.clip(np.std(fit_res), 0.5, 6.0))
    out = dict(hgrid=hgrid, prof_db=prof, unc_db=unc_db,
               ml=(ml_bot, ml_peak, ml_top), bb_amp=bb_amp, slope=slope)

    # Optional temporal smoothing: with PLUVIO_VPR_STATE set to a directory, the
    # profile is blended 50/50 with the previous volume's. Refitting from scratch
    # every 5 minutes makes far-range corrections flicker frame to frame — visible
    # as temporal noise in the served history animation; the profile physically
    # evolves on hour scales, so an EMA loses nothing.
    import os as _os
    state_dir = _os.environ.get("PLUVIO_VPR_STATE")
    if state_dir:
        site = sweeps[0]["site"]
        key = f"vpr_{site[0]:.3f}_{site[1]:.3f}".replace(".", "p").replace("-", "m")
        sp = pathlib.Path(state_dir) / f"{key}.npz"
        try:
            if sp.exists():
                prev = np.load(sp)
                if prev["hgrid"].shape == hgrid.shape:
                    out["prof_db"] = 0.5 * out["prof_db"] + 0.5 * prev["prof_db"]
                    out["unc_db"] = float(0.5 * out["unc_db"] + 0.5 * float(prev["unc_db"]))
            sp.parent.mkdir(parents=True, exist_ok=True)
            tmp = sp.with_suffix(".tmp.npz")
            np.savez(tmp, hgrid=hgrid, prof_db=out["prof_db"], unc_db=out["unc_db"])
            tmp.replace(sp)
        except Exception:
            pass
    return out


def vpr_correction_db(h_km, vpr, sigma_km=None):
    """dB to ADD so a voxel at beam height h represents the rain layer.

    The idealized profile is evaluated through a Gaussian beam of width sigma (km) —
    at 100 km a 1 deg beam is ~1.7 km across, and ignoring that overstates bright-band
    and snow corrections badly. Correction = -P_beam(h), capped at ±VPR_MAX_DB.
    """
    hg, pf = vpr["hgrid"], vpr["prof_db"]
    lin = 10.0 ** (pf / 10.0)
    h = np.asarray(h_km, "float64")
    if sigma_km is None:
        sigma_km = np.full_like(h, 0.3)
    out = np.empty_like(h)
    flat_h = h.ravel(); flat_s = np.clip(np.asarray(sigma_km, "float64").ravel(), 0.05, 2.0)
    flat_o = out.ravel()
    # quantize sigma so the convolution is done once per class, not per voxel
    klass = np.clip(np.round(flat_s / 0.1).astype(int), 1, 20)
    for k in np.unique(klass):
        sig = k * 0.1
        w = np.exp(-0.5 * ((hg[None, :] - hg[:, None]) / sig) ** 2)
        pb = 10.0 * np.log10(np.maximum((w * lin[None, :]).sum(1) / w.sum(1), 1e-9))
        sel = klass == k
        flat_o[sel] = np.interp(np.clip(flat_h[sel], hg[0], hg[-1]), hg, pb)
    corr = np.clip(-out, -VPR_MAX_DB, VPR_MAX_DB)
    unc = np.full_like(h, vpr["unc_db"])
    return corr, unc


def merge_sweeps(sweeps, shape, bounds, polar_to_grid):
    """Merge all sweeps of one radar onto the grid. Returns (dbz, Q_r, h_eff).

    Modes via PLUVIO_SWEEP_MERGE:
      "weighted" — Eqs. 1-2 verbatim: quality-weighted mean in linear Z.
      "best"     — per pixel, the highest-Q_T sweep speaks.
      "lowest"   — the lowest sweep with a valid (clutter-free) voxel speaks; higher
                   sweeps only fill where lower ones have nothing.
      "local" (default) — quality-weighted mean over ONLY the sweeps whose beam sits
                   within LOCAL_DH_M of the lowest usable one at that pixel. Deep rain
                   keeps the multi-sweep intensity stability that made "weighted" win
                   at >=0.5 mm/h; shallow drizzle keeps "lowest"'s immunity to
                   dry-aloft dilution, because higher sweeps fall outside the window.

    Measured on 2026-08-30 0730 (nlhrw wet fraction; old lowest-sweep product = 1.33%):
      weighted 0.63% — dry-aloft sweeps at Q_H ~0.7-0.9 drag wet cells under threshold;
      best     0.46% — worse: Eq. A5 punishes the lowest 500 m by design, so near the
                       radar a higher DRY sweep out-scores the lowest and erases
                       drizzle living below 1 km;
      lowest   0.94-1.24% — matches the old product's behaviour.
    Q_r always combines per Eq. 2; h_eff is the beam height of the lowest usable sweep.
    """
    import os as _os
    mode = _os.environ.get("PLUVIO_SWEEP_MERGE", "local")
    LOCAL_DH_M = float(_os.environ.get("PLUVIO_LOCAL_DH", "800"))

    site = sweeps[0]["site"]
    alt0 = site[2] if len(site) > 2 else 0.0
    vpr = estimate_vpr(sweeps)

    parts = []            # (gz = grid(Q*Z), gq = grid(Q), gh = grid(beam height))
    for sw in sweeps:
        dbz, q_t = process_sweep(sw)
        # Sect. 3.4: extrapolate toward the ground with the fitted idealized profile.
        # Convective voxels (>40 dBZ) are left alone — vertical mixing invalidates the
        # stratiform profile there, which is also what the paper does.
        if vpr is not None:
            h_km = (beam_height_m(sw["rng"], sw["elangle"], alt0)[None, :] - alt0) / 1000.0
            sigma = (sw["rng"][None, :] / 1000.0) * np.radians(1.0) / 2.355   # km
            corr, unc = vpr_correction_db(np.broadcast_to(h_km, dbz.shape), vpr,
                                          np.broadcast_to(sigma, dbz.shape))
            apply = np.isfinite(dbz) & (dbz < VPR_CONV_DBZ)
            dbz = np.where(apply, dbz + corr, dbz)
            q_t = q_t * np.where(apply, q_gauss(corr, C0_DB) * q_gauss(unc, C0_DB), 1.0)
        z_lin = np.where(np.isfinite(dbz), 10.0 ** (dbz / 10.0), 0.0)
        gz = np.nan_to_num(polar_to_grid(q_t * z_lin, sw["az"], sw["rng"], site, shape,
                                         bounds, elangle=sw["elangle"], max_beam_m=1e9), nan=0.0)
        gq = np.nan_to_num(polar_to_grid(q_t, sw["az"], sw["rng"], site, shape, bounds,
                                         elangle=sw["elangle"], max_beam_m=1e9), nan=0.0)
        h_ray = beam_height_m(sw["rng"], sw["elangle"], alt0)
        gh = polar_to_grid(np.broadcast_to(h_ray[None, :], dbz.shape).astype("float32"),
                           sw["az"], sw["rng"], site, shape, bounds,
                           elangle=sw["elangle"], max_beam_m=1e9)
        gh = np.where(np.isfinite(gh), gh, np.inf)
        parts.append((gz, gq, gh))

    num = sum(gz for gz, _, _ in parts)
    den = sum(gq for _, gq, _ in parts)
    one_minus_q = np.ones(shape, "float64")
    for _, gq, _ in parts:
        one_minus_q *= (1.0 - np.clip(gq, 0.0, 1.0))
    # lowest usable sweep per pixel: sweeps are sorted by elevation, first hit wins
    low_z = np.full(shape, np.nan, "float64")
    low_h = np.full(shape, np.inf, "float64")
    for gz, gq, gh in parts:
        fill = (~np.isfinite(low_z)) & (gq > 0)
        low_z = np.where(fill, gz / np.maximum(gq, 1e-12), low_z)
        low_h = np.where(fill, gh, low_h)

    if mode == "weighted":
        z_q = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    elif mode == "best":
        best_q = np.zeros(shape, "float64")
        best_z = np.full(shape, np.nan, "float64")
        for gz, gq, _ in parts:
            take = gq > best_q
            best_z = np.where(take, gz / np.maximum(gq, 1e-12), best_z)
            best_q = np.where(take, gq, best_q)
        z_q = best_z
    elif mode == "lowest":
        z_q = low_z
    else:                                       # "local"
        lnum = np.zeros(shape, "float64")
        lden = np.zeros(shape, "float64")
        for gz, gq, gh in parts:
            sel = (gq > 0) & (gh <= low_h + LOCAL_DH_M)
            lnum += np.where(sel, gz, 0.0)
            lden += np.where(sel, gq, 0.0)
        z_q = np.where(lden > 0, lnum / np.maximum(lden, 1e-12), np.nan)

    dbz_q = np.where(np.isfinite(z_q) & (z_q > 0),
                     10.0 * np.log10(np.maximum(z_q, 1e-12)), np.nan)
    q_r = np.where(den > 0, 1.0 - one_minus_q, 0.0)
    return dbz_q, q_r, np.where(low_h < np.inf, low_h, np.nan)


def gabella(dbz, tr1_db=6.0, n_p=6, tr2=1.3):
    """Sect. 3.6 residual clutter filter on the Cartesian map. Returns clutter mask."""
    import wradlib.classify as classify

    filled = np.nan_to_num(dbz, nan=-32.0)
    mask = classify.filter_gabella(filled, wsize=5, thrsnorain=0.0, tr1=tr1_db,
                                   n_p=n_p, tr2=tr2, rm_nans=False)
    return mask & np.isfinite(dbz)


def dbz_to_rate(dbz):
    """Eq. 3, Marshall-Palmer."""
    z = 10.0 ** (dbz / 10.0)
    return np.where(np.isfinite(dbz), (z / 200.0) ** (1.0 / 1.6), np.nan)


def single_radar(sweeps, shape, bounds, polar_to_grid):
    """Full single-radar chain -> (rate mm/h, Q_r)."""
    rate, q_r, _ = single_radar_h(sweeps, shape, bounds, polar_to_grid)
    return rate, q_r


def single_radar_h(sweeps, shape, bounds, polar_to_grid):
    """Full single-radar chain -> (rate mm/h, Q_r, measurement height m ASL).

    PLUVIO_GABELLA=0 disables the Cartesian Gabella step: its parameters (5x5 window,
    six qualifying pixels, area/circumference 1.3 PIXELS) assume RTCOR's 1 km grid. On
    our ~3 km grid a 15 km drizzle band is five pixels wide and gets classified as
    clutter — measured: the filter removes a third of the wet area (1.26% -> 0.85%) on
    a drizzle day. At this resolution the fuzzy polar QC plus the speckle filter cover
    the same failure mode without the scale mismatch.
    """
    import os as _os
    dbz, q_r, h_eff = merge_sweeps(sweeps, shape, bounds, polar_to_grid)
    if _os.environ.get("PLUVIO_GABELLA", "1") != "0":
        clutter = gabella(dbz)
        dbz = np.where(clutter, np.nan, dbz)
        q_r = np.where(clutter, 0.0, q_r)
    rate = dbz_to_rate(dbz)
    rate = np.where(np.isfinite(dbz), rate, np.where(q_r > 0, 0.0, np.nan))
    return rate, q_r, h_eff


def composite_by_height(per_radar, shape, veto_dh_m=500.0):
    """Cross-radar merge: lowest measurement height wins; veto by peers at that height.

    The chain's own Q cannot arbitrate between radars: Eq. A5 deliberately punishes the
    lowest 500 m, so a distant radar's beam at 0.8 km out-scores the local one at
    0.3 km — measured on the eight-radar Aug-30 composite, both quality-weighted and
    quality-winner merges collapse POD (0.60 / 0.34 against RTCOR's 0.91) because beams
    that sit ABOVE shallow rain dilute or veto the radar that actually samples it.
    Beam height IS monotone in the thing that matters, so this applies the composite-v2
    rule (validated on gauges) to the chain's per-radar products: the radar measuring
    closest to the ground speaks, and only radars measuring within veto_dh_m of it may
    call the cell dry.

    Input: list of (rate, q, h_eff) from single_radar_h. Returns (rate, q).
    """
    rates = np.stack([r for r, _, _ in per_radar])
    qs = np.stack([np.where(np.isfinite(r), q, 0.0) for r, q, _ in per_radar])
    hs = np.stack([np.where(np.isfinite(r) & (q > 0) & np.isfinite(h), h, np.inf)
                   for r, q, h in per_radar])
    pick = np.argmin(hs, axis=0)
    h_win = np.take_along_axis(hs, pick[None], 0)[0]
    out = np.take_along_axis(rates, pick[None], 0)[0]
    q_out = np.take_along_axis(qs, pick[None], 0)[0]
    out = np.where(h_win < np.inf, out, np.nan)
    peers = hs <= (h_win + veto_dh_m)[None]
    n_peer = peers.sum(0)
    wet_peers = (peers & (np.nan_to_num(rates, nan=0.0) > 0.1)).sum(0)
    out = np.where((n_peer > 1) & (wet_peers == 0) & (out > 0.1), 0.0, out)
    return out.astype("float32"), q_out.astype("float32")


def composite_winner(per_radar, shape, consensus_frac=0.5):
    """Winner-takes-all by quality, with a quality-gated consensus.

    The alternative to Eq. 1-2 averaging across RADARS. Weighted averaging dilutes rain
    with legitimate dry-aloft readings from overshooting radars (a distant radar's beam
    at 2 km sees nothing in shallow rain, carries Q~0.6, and halves the wet cell). Here
    the highest-quality radar speaks for the cell, and only radars with comparable
    quality (>= consensus_frac * winner's) get a veto: if all of them say dry, the cell
    is dry. Radars looking far above the winner have low Q and therefore no vote —
    which is exactly the flaw of the naive consensus gate from composite v2, repaired
    with the chain's own quality index.
    """
    rates = np.stack([np.nan_to_num(r, nan=np.nan) for r, _ in per_radar])
    qs = np.stack([np.where(np.isfinite(r), q, 0.0) for r, q in per_radar])
    n = len(per_radar)
    best = np.argmax(qs, axis=0)
    qmax = np.take_along_axis(qs, best[None], 0)[0]
    out = np.take_along_axis(rates, best[None], 0)[0]
    out = np.where(qmax > 0, out, np.nan)
    voters = qs >= np.maximum(consensus_frac * qmax, 1e-6)[None]
    n_vote = voters.sum(0)
    wet_votes = (voters & (np.nan_to_num(rates, nan=0.0) > 0.1)).sum(0)
    out = np.where((n_vote > 1) & (wet_votes == 0) & (out > 0.1), 0.0, out)
    return out.astype("float32"), qmax.astype("float32")


def speckle(rate, min_neighbours=4):
    """Our validated isolated-cell filter (composite v2): rain is spatially coherent."""
    w = (np.nan_to_num(rate, nan=0.0) > 0.1).astype("int8")
    p = np.pad(w, 1)
    neigh = sum(p[i:i + w.shape[0], j:j + w.shape[1]]
                for i in range(3) for j in range(3)) - w
    return np.where(neigh >= min_neighbours, rate,
                    np.where(np.isfinite(rate), 0.0, np.nan))


def composite(per_radar, shape):
    """Quality-weighted composite across radars (Eqs. 1-2 again). Input: [(rate, Q_r)]."""
    num = np.zeros(shape, "float64")
    den = np.zeros(shape, "float64")
    omq = np.ones(shape, "float64")
    for rate, q in per_radar:
        ok = np.isfinite(rate) & (q > 0)
        num += np.where(ok, rate * q, 0.0)
        den += np.where(ok, q, 0.0)
        omq *= np.where(ok, 1.0 - q, 1.0)
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return out.astype("float32"), (1.0 - omq).astype("float32")


def read_sweeps_any(radar: str, stamp: str, max_elangle: float = 6.0):
    """Multi-sweep read for ANY radar, dispatching on which feed carries it.

    KNMI archive (nlhrw, nldhl): full dual-pol, back to 2019.
    DWD opendata (de***): dbzh filtered + unfiltered RhoHV/ZDR, ~2-day window.
    OPERA single-site capture (BE/FR/...): ODIM, DBZH only, 24-h window as captured.
    """
    from tools import knmi_volume as _kv
    from tools import dwd_volume as _dv
    from tools import radar_single_site as _rss

    if radar in _kv.DATASETS:
        path = _kv.fetch(radar, stamp)
        return _kv.read_all_sweeps(path, max_elangle=max_elangle) if path else []
    if radar in _dv.SITES:
        return _dv.read_all_sweeps(radar, stamp)
    path = _rss.find_volume(radar, stamp)
    return _rss.read_all_sweeps(path, max_elangle=max_elangle) if path else []
