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


def test_validate_reports_the_objective_not_just_rmse():
    """Selecting on val RMSE under an FSS+sharpness loss keeps the epoch-1
    model: both v3 arms scored their best RMSE at epoch 1 and then ran 30
    epochs whose objective kept improving. So validate() must report the
    loss it is trained on, an FSS, and the bias the sharpness term inflates.
    """
    import torch

    from model.losses import CombinedLoss
    from model.train import validate

    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 1, 32, 32)
    loader = [(x, y)]

    class Head(torch.nn.Module):
        def forward(self, t):
            return t[:, :1] * 0.5

    loss_fn = CombinedLoss(fss_weight=0.5, sharpness_weight=0.05)
    out = validate(Head(), loader, torch.device("cpu"), 0, loss_fn=loss_fn)
    assert {"val_rmse", "val_loss", "val_fss3", "val_wet_bias"} <= set(out)
    assert out["val_loss"] > 0
    assert 0.0 <= out["val_fss3"] <= 1.0


def test_select_on_resolves_auto_to_the_objective_for_a_shaped_loss():
    """auto = the metric the model is actually minimising."""
    import argparse

    from model.train import resolve_select_on

    def ns(**kw):
        base = {"select_on": "auto", "fss_weight": 0.0, "sharpness_weight": 0.0,
                "quantiles": None}
        base.update(kw)
        return argparse.Namespace(**base)

    assert resolve_select_on(ns()) == "val_rmse"
    assert resolve_select_on(ns(fss_weight=0.5)) == "val_loss"
    assert resolve_select_on(ns(sharpness_weight=0.05)) == "val_loss"
    assert resolve_select_on(ns(quantiles="0.1,0.5,0.9")) == "val_loss"
    assert resolve_select_on(ns(fss_weight=0.5, select_on="val_rmse")) == "val_rmse"


def test_two_epoch_run_selects_on_the_objective_and_records_it(synthetic_store, tmp_path):
    """End-to-end: a shaped loss must be able to pick a LATER epoch than the
    RMSE-best one, and the checkpoint has to say which metric chose it. Cheap
    guard against a typo in the selection block costing a 12-hour GPU run."""
    import torch

    from model.train import main

    ckpt = tmp_path / "smoke.pt"
    rc = main([
        "--zarr", str(synthetic_store),
        "--epochs", "2", "--batch-size", "2", "--base-channels", "4",
        "--max-train-samples", "8", "--max-val-samples", "4", "--num-workers", "0",
        "--fss-weight", "0.5", "--sharpness-weight", "0.05",
        "--patience", "5", "--device", "cpu",
        "--checkpoint", str(ckpt),
    ])
    assert rc == 0
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["select_on"] == "val_loss"
    assert saved["val_loss"] is not None
    assert "val_fss3" in saved and "val_wet_bias" in saved


def test_companion_metrics_cover_skill_and_rmse_for_a_shaped_loss():
    import argparse

    from model.train import _companion_metrics

    def ns(save_best_of="auto"):
        return argparse.Namespace(save_best_of=save_best_of)

    assert _companion_metrics(ns(), "val_loss") == ["val_fss3", "val_rmse"]
    # never a companion for the metric that already drives selection
    assert _companion_metrics(ns(), "val_fss3") == ["val_rmse"]
    # plain RMSE selection means the loss is RMSE: nothing else to compare
    assert _companion_metrics(ns(), "val_rmse") == []
    assert _companion_metrics(ns("val_fss3"), "val_loss") == ["val_fss3"]
    assert _companion_metrics(ns(""), "val_loss") == []


def test_companion_checkpoint_is_written_next_to_the_primary(synthetic_store, tmp_path):
    """A run under val_loss selection must also leave the best-FSS weights on
    disk, so the benchmark can score both instead of trusting the loss to pick
    the better forecast."""
    import torch

    from model.train import main

    ckpt = tmp_path / "smoke.pt"
    rc = main([
        "--zarr", str(synthetic_store),
        "--epochs", "2", "--batch-size", "2", "--base-channels", "4",
        "--max-train-samples", "8", "--max-val-samples", "4", "--num-workers", "0",
        "--fss-weight", "0.5", "--sharpness-weight", "0.05",
        "--patience", "5", "--device", "cpu",
        "--checkpoint", str(ckpt),
    ])
    assert rc == 0
    companion = tmp_path / "smoke.val_fss3.pt"
    assert companion.exists()
    saved = torch.load(companion, map_location="cpu", weights_only=False)
    assert saved["select_on"] == "val_fss3"
    assert saved["val_fss3"] is not None


def test_a_stopped_run_resumes_where_it_left_off(synthetic_store, tmp_path):
    """Pause and resume: the state file has to carry the optimizer, the LR
    schedule and the early-stopping counters, so stopping the run (a reboot,
    or the machine wanted for something else) costs at most the unfinished
    epoch — not the run."""
    import torch

    from model.train import main

    ckpt = tmp_path / "resume.pt"
    common = [
        "--zarr", str(synthetic_store), "--batch-size", "2", "--base-channels", "4",
        "--max-train-samples", "8", "--max-val-samples", "4", "--num-workers", "0",
        "--fss-weight", "0.5", "--patience", "5", "--device", "cpu",
        "--checkpoint", str(ckpt),
    ]
    assert main([*common, "--epochs", "2"]) == 0
    state_path = tmp_path / "resume.state.pt"
    assert state_path.exists()
    first = torch.load(state_path, map_location="cpu", weights_only=False)
    assert first["epoch"] == 2
    assert {"optimizer", "scheduler", "best_score", "no_improve"} <= set(first)

    # Resuming with a larger budget continues from epoch 3, it does not restart
    assert main([*common, "--epochs", "4"]) == 0
    second = torch.load(state_path, map_location="cpu", weights_only=False)
    assert second["epoch"] == 4
    # and the optimizer state moved on rather than being reinitialised
    assert second["optimizer"]["state"], "resumed run has no optimizer moments"


def test_resume_never_starts_from_scratch(synthetic_store, tmp_path):
    import torch

    from model.train import main

    ckpt = tmp_path / "fresh.pt"
    common = [
        "--zarr", str(synthetic_store), "--batch-size", "2", "--base-channels", "4",
        "--max-train-samples", "8", "--max-val-samples", "4", "--num-workers", "0",
        "--patience", "5", "--device", "cpu", "--checkpoint", str(ckpt),
    ]
    assert main([*common, "--epochs", "2"]) == 0
    assert main([*common, "--epochs", "1", "--resume", "never"]) == 0
    state = torch.load(tmp_path / "fresh.state.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == 1          # restarted, not continued from 2


def test_a_signal_mid_epoch_saves_a_resumable_state(synthetic_store, tmp_path, monkeypatch):
    """SIGTERM inside an epoch must not lose the run: the partial epoch is
    discarded (its metrics are not comparable) and the state records the last
    completed epoch."""
    import signal as signal_mod

    import torch

    import model.train as tr

    real = tr.train_one_epoch
    calls = {"n": 0}

    def fake(*a, should_stop=None, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            # simulate the signal arriving during the second epoch
            import os
            os.kill(os.getpid(), signal_mod.SIGTERM)
        return real(*a, should_stop=should_stop, **kw)

    monkeypatch.setattr(tr, "train_one_epoch", fake)
    ckpt = tmp_path / "sig.pt"
    rc = tr.main([
        "--zarr", str(synthetic_store), "--epochs", "6", "--batch-size", "2",
        "--base-channels", "4", "--max-train-samples", "8", "--max-val-samples", "4",
        "--num-workers", "0", "--patience", "5", "--device", "cpu",
        "--checkpoint", str(ckpt),
    ])
    assert rc == 0
    state = torch.load(tmp_path / "sig.state.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] in (1, 2)     # the last epoch whose metrics were real
    assert calls["n"] == 2              # it stopped instead of running all six
