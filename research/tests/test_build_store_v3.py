"""tools.build_store_v3: the explicit name->extent table (1.12) and the
issue_time units guard, both raise loudly rather than guessing/coercing."""

from __future__ import annotations

import numpy as np
import pytest
from tools import build_store_v3 as bsv3


def test_extent_for_known_names():
    assert bsv3._extent_for("radar") == bsv3._TRIMMED
    assert bsv3._extent_for("truth") == bsv3._TRIMMED
    assert bsv3._extent_for("sst") == bsv3._UNTRIMMED
    assert bsv3._extent_for("msg_ir108") == bsv3._UNTRIMMED
    assert bsv3._extent_for("alaro_precip") == bsv3._UNTRIMMED
    assert bsv3._extent_for("aws_temp") == bsv3._UNTRIMMED


def test_extent_for_unknown_name_raises():
    with pytest.raises(ValueError, match="extent table"):
        bsv3._extent_for("mystery_channel")


def test_assert_epoch_seconds_accepts_real_seconds():
    t = np.array([1_700_000_000, 1_700_001_800], dtype="int64")
    bsv3._assert_epoch_seconds(t, "test")  # must not raise


def test_assert_epoch_seconds_rejects_milliseconds():
    t = np.array([1_700_000_000_000, 1_700_001_800_000], dtype="int64")
    with pytest.raises(ValueError, match="epoch seconds"):
        bsv3._assert_epoch_seconds(t, "test")


def test_assert_epoch_seconds_warns_but_does_not_raise_on_zero(caplog):
    # A zero-filled slot (e.g. a resize-before-write crash window) is not a
    # units mixup — raising here would take down every run over one bad slot.
    t = np.array([0, 1_700_000_000], dtype="int64")
    with caplog.at_level("WARNING"):
        bsv3._assert_epoch_seconds(t, "test")  # must not raise
    assert any("1 issue_time slot" in r.message for r in caplog.records)


def test_assert_epoch_seconds_empty_is_noop():
    bsv3._assert_epoch_seconds(np.array([], dtype="int64"), "test")  # must not raise
