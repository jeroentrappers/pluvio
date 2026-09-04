"""Quantile models in the frozen benchmark (2.2): median for the point
metrics, CRPS from the quantile stack, reliability from P(rate > thr)."""

from __future__ import annotations

import numpy as np
import pytest

from tools import benchmark as bm
from tools._stats import RELIABILITY_BINS, SampleStats


def test_crps_from_quantiles_is_between_zero_and_mae_of_median_for_a_sharp_forecast():
    obs = np.array([1.0, 2.0, 3.0])
    q = np.stack([obs - 0.5, obs, obs + 0.5])          # perfectly centred, spread 0.5
    total = bm.crps_sum_from_quantiles(q, obs, (0.1, 0.5, 0.9))
    # per cell: pinball(0.1)=0.1*0.5=0.05, pinball(0.5)=0, pinball(0.9)=0.05 -> mean 0.0333 -> x2 = 0.0667
    assert total == pytest.approx(3 * 2 * (0.05 + 0.0 + 0.05) / 3)
    wide = np.stack([obs - 5, obs, obs + 5])
    assert bm.crps_sum_from_quantiles(wide, obs, (0.1, 0.5, 0.9)) > total   # wider spread scores worse


def test_reliability_stats_bin_probabilities_and_count_exceedances():
    obs = np.array([0.0, 0.0, 2.0, 2.0])
    # cell 0/1: dry with all quantiles 0 -> P=0 (bin 0); cell 2/3: q=(1,2,3) -> P(>2.0)=0.5 (bin 5)
    q = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 3.0, 3.0]])
    rel = bm.reliability_stats_from_quantiles(q, obs, (0.1, 0.5, 0.9), [2.0])
    cnt, ob, ps = rel[2.0]
    assert cnt.sum() == 4 and cnt[0] == 2 and cnt[5] == 2
    assert ob[5] == 2 and ob[0] == 0
    assert ps[5] == pytest.approx(1.0)


def test_sample_stats_aggregate_reports_crps_and_reliability():
    st = SampleStats([1.0], [1])
    rel = {1.0: (np.array([2.0] + [0] * 8 + [2.0]), np.array([0.0] * 9 + [2.0]), np.array([0.1] + [0] * 8 + [1.9]))}
    st.add(issue_epoch=0, n=4, sum_e=0.0, sum_abs_e=4.0, sum_sq_e=4.0,
           cat={1.0: (2, 0, 0)}, fss={1.0: {1: (0.0, 1.0)}}, sum_crps=1.0, rel=rel)
    row = next(iter(st.aggregate().values()))
    assert row["mae"] == pytest.approx(1.0) and row["crps"] == pytest.approx(0.25)
    r = row["reliability"]
    assert r is not None and r["count"][0] == 2 and r["count"][-1] == 2
    assert r["observed_freq"][-1] == pytest.approx(1.0) and r["forecast_prob"][-1] == pytest.approx(0.95)
    assert r["observed_freq"][3] is None


def test_deterministic_records_keep_crps_equal_to_mae():
    st = SampleStats([1.0], [1])
    st.add(issue_epoch=0, n=2, sum_e=1.0, sum_abs_e=3.0, sum_sq_e=5.0,
           cat={1.0: (1, 0, 1)}, fss={1.0: {1: (0.0, 1.0)}})
    row = next(iter(st.aggregate().values()))
    assert row["crps"] == row["mae"] == pytest.approx(1.5)
    assert row["reliability"] is None
    assert RELIABILITY_BINS == 10
