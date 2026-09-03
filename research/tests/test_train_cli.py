"""CLI argument validation for model.train (WP 2.1b)."""

from __future__ import annotations

import pytest

from model.train import main


def test_sharpness_weight_above_cap_is_rejected():
    # The review measured that noise injection is locally rewarded up to
    # parity with the target's gradient energy, and that the crossover
    # moves past a higher weight above ~0.3 (model/losses.py:sharpness_loss,
    # "Residual gaming risk"). Anything above that cap must be rejected
    # before training starts, not silently accepted.
    with pytest.raises(SystemExit, match=r"--sharpness-weight must be <= 0\.3"):
        main(["--sharpness-weight", "0.31"])


def test_sharpness_weight_at_cap_is_accepted_by_validation():
    # 0.3 itself is still allowed; parsing should get past the cap check
    # (and fail later, on the missing --data/--zarr, not on the weight).
    with pytest.raises(SystemExit) as excinfo:
        main(["--sharpness-weight", "0.3"])
    assert "--sharpness-weight" not in str(excinfo.value)


def test_negative_sharpness_weight_is_still_rejected():
    with pytest.raises(SystemExit, match=r"--sharpness-weight must be >= 0"):
        main(["--sharpness-weight", "-0.1"])


def test_lagrangian_channels_rejected_on_the_legacy_hdf5_dataset(tmp_path):
    # The Lagrangian channels (2.3) are assembled in
    # ZarrCorrectionDataset.build_input; the legacy radar-HDF5 dataset never
    # goes through it, so asking for them there must fail before training
    # rather than silently training a plain-channel model.
    with pytest.raises(SystemExit, match=r"--lagrangian-channels needs the zarr store"):
        main(["--data", str(tmp_path), "--lagrangian-channels", "1"])


def test_lagrangian_channels_rejects_out_of_range_value(capsys):
    with pytest.raises(SystemExit):
        main(["--lagrangian-channels", "3"])
    assert "--lagrangian-channels" in capsys.readouterr().err
