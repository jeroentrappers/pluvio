"""Spatial gauge adjustment of radar accumulations — RTCOR Appendix B, verbatim.

The decomposition showed the radar chain is most of RTCOR's edge, but the adjustment
still adds up to ~0.09 CSI at higher thresholds, and it is fully specified (Overeem et
al. 2025, Appendix B), so it is replicated exactly:

    F_adj(x) piecewise from S_w,r / S_w,g with floor T = 0.25 mm         (B1)
    S_w,X = sum_n w_n R_X,n over radar-gauge pairs                        (B2)
    w_n = (G(d, r_s) + v G(d, r_l)) / (1 + v) * Q_r,n * Q_g,n             (B3)
    G truncated Gaussian, exp(-4 d^2/r_d^2) rescaled to hit 0 at r_d      (B4)
    R_adj = R / 10 ** (F_adj_db / 10)                                     (B5)

with r_l = 500 km, v = 0.1, gauge quality fixed at 0.9. r_s is "the range of a
seasonally-dependent variogram of 1 h rainfall" (Van de Beek et al. 2012); the summer
value of ~30 km is used here and must be revisited for winter.

⚠️ EVALUATION HYGIENE. Adjusting with the same gauges the product is scored against is
in-sample by construction — RTCOR itself has this property when scored on KNMI
stations. Honest scoring of an adjusted product here means: adjust with NL gauges,
score on BE gauges (or vice versa), or leave the scored station out of B2. The
adjustment uses the PREVIOUS clock-hour (the paper's own latency), which weakens but
does not remove the circularity.
"""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger("pluvio.gauge_adjust")

import os

T_MM = 0.25
R_S_KM = float(os.environ.get("PLUVIO_GADJ_RS_KM", "30.0"))
R_L_KM = 500.0
V_LONG = 0.1
Q_GAUGE = 0.9


def _gaussian_w(d_km, r_d):
    """Eq. B4: truncated Gaussian reaching exactly 0 at r_d."""
    g = (np.exp(-4.0 * (d_km / r_d) ** 2) - np.exp(-4.0)) / (1.0 - np.exp(-4.0))
    return np.where(d_km <= r_d, np.clip(g, 0.0, 1.0), 0.0)


def adjustment_field(gauge_acc, radar_acc_at_gauges, gauge_lat, gauge_lon,
                     radar_quality_at_gauges, bounds, shape):
    """F_adj in dB on the analysis grid, from one hour's radar-gauge pairs.

    gauge_acc / radar_acc_at_gauges: 60-min accumulations (mm) per gauge.
    """
    w, s, e, n = bounds
    h, wd = shape
    glat = np.asarray(gauge_lat, "float64")
    glon = np.asarray(gauge_lon, "float64")
    lat = np.linspace(n, s, h)[:, None]
    lon = np.linspace(w, e, wd)[None, :]

    sw_r = np.zeros(shape)
    sw_g = np.zeros(shape)
    for gl, go, rg, rr, qr in zip(glat, glon, np.asarray(gauge_acc),
                                  np.asarray(radar_acc_at_gauges),
                                  np.asarray(radar_quality_at_gauges)):
        if not (np.isfinite(rg) and np.isfinite(rr)):
            continue
        dx = (lon - go) * 111.32 * np.cos(np.radians(lat))
        dy = (lat - gl) * 111.32
        d = np.sqrt(dx ** 2 + dy ** 2)
        wn = (_gaussian_w(d, R_S_KM) + V_LONG * _gaussian_w(d, R_L_KM)) / (1.0 + V_LONG)
        wn = wn * np.clip(qr, 0.0, 1.0) * Q_GAUGE
        sw_r += wn * rr
        sw_g += wn * rg

    # Eq. B1 with the floor T on BOTH sums
    num = np.where(sw_r > T_MM, sw_r, T_MM)
    den = np.where(sw_g > T_MM, sw_g, T_MM)
    factor = np.where((sw_r <= T_MM) & (sw_g <= T_MM), 1.0, num / den)
    return (10.0 * np.log10(np.clip(factor, 1e-3, 1e3))).astype("float32")


def apply(rate, f_adj_db):
    """Eq. B5 on a rate field (linear in accumulation, so the same factor applies)."""
    return rate / np.power(10.0, np.nan_to_num(f_adj_db, nan=0.0) / 10.0)
