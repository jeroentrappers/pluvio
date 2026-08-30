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
    """Eq. A5."""
    return sigmoid(h_km, 0.0, H_L) - 0.05 / 0.95 * sigmoid(h_km, H_H, H_M)


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
VPR_MAX_DB = 10.0      # cap on the applied correction
VPR_CONV_DBZ = 40.0    # voxels above this are convective: no VPR correction (Sect. 3.4)


def estimate_vpr(sweeps, r_min_km=20.0, r_max_km=120.0, dz_km=0.25, top_km=6.0):
    """Apparent VPR from the volume itself: mean dBZ per height bin over all sweeps.

    Simplified from Hazenberg et al. (2013, 2014): the paper fits idealized stratiform /
    undefined profiles with a polarimetric melting-layer detection; this takes the mean
    apparent profile in an annulus where the beam is narrow enough to resolve structure,
    which captures the two errors that matter most — bright-band inflation and the
    fall-off into ice above it. Returns (heights_km, profile_db, spread_db) or None.
    """
    alt = sweeps[0]["site"][2] if len(sweeps[0]["site"]) > 2 else 0.0
    edges = np.arange(0.0, top_km + dz_km, dz_km)
    sums = np.zeros(len(edges) - 1)
    sq = np.zeros(len(edges) - 1)
    cnt = np.zeros(len(edges) - 1)
    for sw in sweeps:
        rng = sw["rng"]
        sel = (rng >= r_min_km * 1000.0) & (rng <= r_max_km * 1000.0)
        if not sel.any():
            continue
        h_km = (beam_height_m(rng[sel], sw["elangle"], alt) - alt) / 1000.0
        dbz = sw["dbz"][:, sel]
        ok = np.isfinite(dbz) & (dbz > 0.0) & (dbz < VPR_CONV_DBZ)
        if not ok.any():
            continue
        idx = np.clip(np.digitize(h_km, edges) - 1, 0, len(cnt) - 1)
        for j in range(len(cnt)):
            col = ok[:, idx == j]
            v = dbz[:, idx == j][col]
            if v.size:
                sums[j] += v.sum()
                sq[j] += (v ** 2).sum()
                cnt[j] += v.size
    if (cnt > 50).sum() < 4:
        return None
    prof = np.where(cnt > 50, sums / np.maximum(cnt, 1), np.nan)
    # Uncertainty of the PROFILE ESTIMATE — the standard error of the bin mean — not the
    # field's natural spatial variability. Confusing the two multiplies quality by
    # q_gauss(~9 dB / 3 dB) ~ 0.002 everywhere and silently zeroes the whole product
    # (measured: Q_r fell from 0.72 to 0.004 before this distinction was made).
    std = np.sqrt(np.maximum(sq / np.maximum(cnt, 1)
                             - (sums / np.maximum(cnt, 1)) ** 2, 0.0))
    stderr = np.where(cnt > 50, std / np.sqrt(np.maximum(cnt, 1)), np.nan)
    mids = (edges[:-1] + edges[1:]) / 2.0
    return mids, prof, stderr


def vpr_correction_db(h_km, vpr):
    """dB to ADD to a voxel at height h so it represents the reference height."""
    mids, prof, spread = vpr
    ok = np.isfinite(prof)
    if ok.sum() < 3:
        return np.zeros_like(h_km), np.zeros_like(h_km)
    ref = np.interp(VPR_REF_KM, mids[ok], prof[ok])
    at_h = np.interp(np.clip(h_km, mids[ok].min(), mids[ok].max()), mids[ok], prof[ok])
    corr = np.clip(ref - at_h, -VPR_MAX_DB, VPR_MAX_DB)
    unc = np.interp(np.clip(h_km, mids[ok].min(), mids[ok].max()),
                    mids[ok], np.nan_to_num(spread[ok], nan=3.0))
    return corr, unc


def merge_sweeps(sweeps, shape, bounds, polar_to_grid):
    """Quality-weighted merge of all sweeps of one radar onto the grid (Eqs. 1-2).

    Averaging is done in linear Z, as reflectivity should be. Returns (dbz, Q_r).
    """
    num = np.zeros(shape, "float64")
    den = np.zeros(shape, "float64")
    one_minus_q = np.ones(shape, "float64")
    site = sweeps[0]["site"]
    vpr = estimate_vpr(sweeps)
    for sw in sweeps:
        dbz, q_t = process_sweep(sw)
        # Sect. 3.4: extrapolate to the ground with the volume's own apparent profile.
        # Convective voxels (>40 dBZ) are left alone — vertical mixing invalidates the
        # stratiform profile there, which is also what the paper does.
        if vpr is not None:
            alt = site[2] if len(site) > 2 else 0.0
            h_km = (beam_height_m(sw["rng"], sw["elangle"], alt)[None, :] - alt) / 1000.0
            corr, unc = vpr_correction_db(np.broadcast_to(h_km, dbz.shape), vpr)
            apply = np.isfinite(dbz) & (dbz < VPR_CONV_DBZ)
            dbz = np.where(apply, dbz + corr, dbz)
            q_t = q_t * np.where(apply, q_gauss(corr, C0_DB) * q_gauss(unc, C0_DB), 1.0)
        z_lin = np.where(np.isfinite(dbz), 10.0 ** (dbz / 10.0), 0.0)
        # grid Q*Z and Q separately so the ratio is a true weighted mean per cell
        gz = polar_to_grid(q_t * z_lin, sw["az"], sw["rng"], site, shape, bounds,
                           elangle=sw["elangle"], max_beam_m=1e9)
        gq = polar_to_grid(q_t, sw["az"], sw["rng"], site, shape, bounds,
                           elangle=sw["elangle"], max_beam_m=1e9)
        gz = np.nan_to_num(gz, nan=0.0)
        gq = np.nan_to_num(gq, nan=0.0)
        num += gz
        den += gq
        one_minus_q *= (1.0 - np.clip(gq, 0.0, 1.0))
    z_q = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    dbz_q = np.where(np.isfinite(z_q) & (z_q > 0), 10.0 * np.log10(np.maximum(z_q, 1e-12)), np.nan)
    q_r = np.where(den > 0, 1.0 - one_minus_q, 0.0)
    return dbz_q, q_r


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
    dbz, q_r = merge_sweeps(sweeps, shape, bounds, polar_to_grid)
    clutter = gabella(dbz)
    dbz = np.where(clutter, np.nan, dbz)
    q_r = np.where(clutter, 0.0, q_r)
    rate = dbz_to_rate(dbz)
    rate = np.where(np.isfinite(dbz), rate, np.where(q_r > 0, 0.0, np.nan))
    return rate, q_r


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
