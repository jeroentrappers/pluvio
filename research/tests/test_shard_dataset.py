"""Pre-rendered training shards (WP 2.6): tools/render_shards.py + model/shard_dataset.py.

The whole point of the shard path is that the loss curve is unchanged, so the
central assertions here are equality assertions: rendered sample *i* is
``ZarrCorrectionDataset[i]``, with the float16 cast applied identically on both
sides (``shard_dataset.cast_for_shard`` — the renderer and these tests call the
same function, so "same cast" is a property of the code, not a convention).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch

from model.shard_dataset import (
    INDEX_NAME,
    MANIFEST_NAME,
    ShardDataset,
    ShardRecipeMismatch,
    ShardStoreIncomplete,
    cast_for_shard,
    compare_recipes,
    recipe_from_dataset,
)
from model.zarr_dataset import ZarrCorrectionDataset, issue_time_split
from tools import render_shards

_DT_MIN = render_shards._DT_MIN
_DT_MAX = render_shards._DT_MAX

# The synthetic store carries leads 0/30/60/90; lead 0 is the analysis, not a
# forecast, so the trainable set is the non-zero leads.
LEADS = "30,60,90"


def _render(store, out, *, split="all", dtype="float16", extra=()):
    argv = ["--zarr", str(store), "--out", str(out), "--split", split,
            "--dtype", dtype, "--leads", LEADS, "--samples-per-shard", "5",
            *extra]
    assert render_shards.main(argv) == 0
    return out / split


def _zarr_set(store, *, split="all", val_frac=0.2, require_rain_fraction=None,
              lagrangian_channels=0):
    time_range = None
    if split != "all":
        boundary = issue_time_split(store, val_frac)
        time_range = ((_DT_MIN, boundary) if split == "train" else (boundary, _DT_MAX))
    return ZarrCorrectionDataset(
        store, time_range=time_range,
        leads_min=tuple(int(v) for v in LEADS.split(",")),
        require_rain_fraction=require_rain_fraction,
        lagrangian_channels=lagrangian_channels,
    )


# ───────────────────────────────────────────────────────── equality (the core)


def test_rendered_samples_are_bit_for_bit_the_zarr_samples_float16(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    zds = _zarr_set(synthetic_store)
    sds = ShardDataset(shard_dir)

    assert len(sds) == len(zds) > 0
    assert sds.n_channels == zds.n_channels
    assert sds.grid_hw == zds.grid_hw

    for i in range(len(zds)):
        zx, zy = zds[i]
        sx, sy = sds[i]
        # float16 is where the shards live; the reference gets the identical
        # cast, then both come back as float32 tensors.
        want_x = torch.from_numpy(cast_for_shard(zx.numpy(), "float16").astype("float32"))
        want_y = torch.from_numpy(cast_for_shard(zy.numpy(), "float16").astype("float32"))
        assert sx.dtype is torch.float32 and sy.dtype is torch.float32
        assert torch.equal(sx, want_x), f"input mismatch at sample {i}"
        assert torch.equal(sy, want_y), f"target mismatch at sample {i}"


def test_float32_shards_are_bit_for_bit_the_unquantised_zarr_samples(synthetic_store, tmp_path):
    # With --dtype float32 there is no quantisation at all, so equality holds
    # against the raw dataset output — proof the only difference in the
    # float16 case is the documented cast.
    shard_dir = _render(synthetic_store, tmp_path / "shards32", dtype="float32")
    zds = _zarr_set(synthetic_store)
    sds = ShardDataset(shard_dir)
    for i in range(len(zds)):
        zx, zy = zds[i]
        sx, sy = sds[i]
        assert torch.equal(sx, zx) and torch.equal(sy, zy), f"sample {i}"


def test_float16_cast_actually_changes_values(synthetic_store, tmp_path):
    # Guards the test above from being vacuous: if the normalised channels all
    # happened to be float16-exact, the two equality tests would be the same
    # test. The lead/120 plane alone is not representable.
    zds = _zarr_set(synthetic_store)
    x, _ = zds[0]
    assert not np.array_equal(cast_for_shard(x.numpy(), "float16").astype("float32"), x.numpy())


def test_sample_order_and_metadata_match_the_zarr_index(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    zds = _zarr_set(synthetic_store)
    meta = np.load(shard_dir / INDEX_NAME)
    assert [int(v) for v in meta["issue_epoch"]] == [s.issue_epoch for s in zds.index]
    assert [int(v) for v in meta["lead_min"]] == [s.lead_min for s in zds.index]
    assert [int(v) for v in meta["target_idx"]] == [s.target_idx for s in zds.index]


def test_shard_boundaries_never_split_an_issue(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    manifest = json.loads((shard_dir / MANIFEST_NAME).read_text())
    meta = np.load(shard_dir / INDEX_NAME)
    seen: set[int] = set()
    for shard in manifest["shards"]:
        lo = shard["first_sample"]
        issues = {int(v) for v in meta["issue_idx"][lo:lo + shard["n_samples"]]}
        assert not (issues & seen), "an issue's samples were split across shards"
        seen |= issues
    assert len(manifest["shards"]) > 1, "test store must produce several shards"


def test_train_val_splits_match_the_zarr_split(synthetic_store, tmp_path):
    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train,val")

    for split in ("train", "val"):
        zds = _zarr_set(synthetic_store, split=split)
        sds = ShardDataset(root / split)
        assert len(sds) == len(zds) > 0
        meta = np.load(root / split / INDEX_NAME)
        assert [int(v) for v in meta["issue_epoch"]] == [s.issue_epoch for s in zds.index]
        zx, zy = zds[0]
        sx, sy = sds[0]
        assert torch.equal(sx, torch.from_numpy(
            cast_for_shard(zx.numpy(), "float16").astype("float32")))
        assert torch.equal(sy, torch.from_numpy(
            cast_for_shard(zy.numpy(), "float16").astype("float32")))

    train_epochs = {int(v) for v in np.load(root / "train" / INDEX_NAME)["issue_epoch"]}
    val_epochs = {int(v) for v in np.load(root / "val" / INDEX_NAME)["issue_epoch"]}
    assert not (train_epochs & val_epochs)
    boundary = int(issue_time_split(synthetic_store, 0.2).timestamp())
    assert max(train_epochs) < boundary <= min(val_epochs)


def test_manifest_records_the_recipe_grid_counts_and_source_hash(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    manifest = json.loads((shard_dir / MANIFEST_NAME).read_text())
    zds = _zarr_set(synthetic_store)

    assert manifest["complete"] is True
    assert manifest["n_samples"] == len(zds)
    assert manifest["grid_hw"] == list(zds.grid_hw)
    assert manifest["n_channels"] == zds.n_channels
    # channel_recipe is the dataset's own recipe (ds.channel_recipe()), the one
    # train.py puts in the checkpoint — not a second, drifting copy here.
    assert manifest["channel_recipe"] == zds.channel_recipe()
    names = manifest["channel_recipe"]["channel_names"]
    assert len(names) == zds.n_channels
    assert names[zds.history_steps] == "nowcast_at_lead"
    assert manifest["recipe"]["aux_channels"] == list(zds.aux_channels)
    assert manifest["recipe"]["static_channels"] == list(zds.static_channels)
    assert manifest["recipe"]["normalise_version"] >= 1
    assert manifest["source_store"]["hash"]
    assert manifest["source_store"]["n_issues"] == len(zds._issue_epoch)
    for shard in manifest["shards"]:
        assert len(shard["sha256_x"]) == 64 and len(shard["sha256_y"]) == 64


def test_source_store_hash_changes_when_the_store_changes(synthetic_store, tmp_path):
    import zarr

    before = render_shards.source_store_hash(synthetic_store)["hash"]
    root = zarr.open_group(str(synthetic_store), mode="a")
    t = root["issue_time"][:]
    t[-1] = int(t[-1]) + 1800
    root["issue_time"][:] = t
    assert render_shards.source_store_hash(synthetic_store)["hash"] != before


# ────────────────────────────────────────────────────────────────── resume


def test_render_is_resumable_after_a_kill(synthetic_store, tmp_path, monkeypatch):
    reference = _render(synthetic_store, tmp_path / "ref")
    ref_manifest = json.loads((reference / MANIFEST_NAME).read_text())

    out = tmp_path / "resumed"
    real = render_shards._render_shard
    calls = {"n": 0}

    def die_after_two(job):
        if calls["n"] >= 2:
            raise KeyboardInterrupt("simulated kill mid-render")
        calls["n"] += 1
        return real(job)

    monkeypatch.setattr(render_shards, "_render_shard", die_after_two)
    with pytest.raises(KeyboardInterrupt):
        _render(synthetic_store, out)

    partial = json.loads((out / "all" / MANIFEST_NAME).read_text())
    assert partial["complete"] is False
    assert len(partial["shards"]) == 2 < len(ref_manifest["shards"])
    # An incomplete store must never quietly train on its first two shards.
    with pytest.raises(ShardStoreIncomplete, match="incomplete"):
        ShardDataset(out / "all")

    monkeypatch.setattr(render_shards, "_render_shard", real)
    shard_dir = _render(synthetic_store, out)
    resumed = json.loads((shard_dir / MANIFEST_NAME).read_text())
    assert resumed["complete"] is True
    assert resumed["n_samples"] == ref_manifest["n_samples"]
    # Byte-identical to a clean render, including the shards kept from the
    # killed run.
    assert [(s["id"], s["sha256_x"], s["sha256_y"]) for s in resumed["shards"]] == \
           [(s["id"], s["sha256_x"], s["sha256_y"]) for s in ref_manifest["shards"]]
    ShardDataset(shard_dir, verify_checksums=True)


def test_resume_keeps_existing_shards_and_renders_only_the_rest(synthetic_store, tmp_path,
                                                                monkeypatch):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    (shard_dir / "x_00001.npy").unlink()

    rendered: list[int] = []
    real = render_shards._render_shard

    def spy(job):
        rendered.append(job["id"])
        return real(job)

    monkeypatch.setattr(render_shards, "_render_shard", spy)
    _render(synthetic_store, tmp_path / "shards")
    assert rendered == [1], "only the deleted shard should be re-rendered"
    ShardDataset(shard_dir, verify_checksums=True)


def test_corrupt_shard_is_caught_by_checksum_verification(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    path = shard_dir / "x_00000.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ShardStoreIncomplete, match="sha256"):
        ShardDataset(shard_dir, verify_checksums=True)


def test_missing_shard_file_raises(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    (shard_dir / "y_00000.npy").unlink()
    with pytest.raises(ShardStoreIncomplete, match="missing shard file"):
        ShardDataset(shard_dir)


def test_parallel_render_is_byte_identical_to_a_serial_one(synthetic_store, tmp_path):
    serial = _render(synthetic_store, tmp_path / "serial")
    parallel = _render(synthetic_store, tmp_path / "parallel", extra=["--workers", "3"])
    keys = ("id", "n_samples", "first_sample", "sha256_x", "sha256_y")
    a = [{k: s[k] for k in keys} for s in json.loads((serial / MANIFEST_NAME).read_text())["shards"]]
    b = [{k: s[k] for k in keys}
         for s in json.loads((parallel / MANIFEST_NAME).read_text())["shards"]]
    assert a == b
    ShardDataset(parallel, verify_checksums=True)


# ───────────────────────────────────────────────────────── recipe mismatch


def test_expected_recipe_mismatch_raises(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    zds = _zarr_set(synthetic_store)
    expected = recipe_from_dataset(zds, split="all", dtype="float16",
                                   split_boundary_epoch=None, val_frac=0.2)
    ShardDataset(shard_dir, expected_recipe=expected)      # matches → fine

    expected["leads_min"] = [30, 60]
    with pytest.raises(ShardRecipeMismatch, match="leads_min"):
        ShardDataset(shard_dir, expected_recipe=expected)


def test_bumped_normalise_version_invalidates_shards(synthetic_store, tmp_path, monkeypatch):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    zds = _zarr_set(synthetic_store)
    monkeypatch.setattr("model.zarr_dataset.NORMALISE_VERSION", 99)
    expected = recipe_from_dataset(zds, split="all", dtype="float16",
                                   split_boundary_epoch=None, val_frac=0.2)
    with pytest.raises(ShardRecipeMismatch, match="normalise_version"):
        ShardDataset(shard_dir, expected_recipe=expected)


def test_rerender_with_a_different_recipe_into_the_same_dir_raises(synthetic_store, tmp_path):
    out = tmp_path / "shards"
    _render(synthetic_store, out)
    with pytest.raises(ShardRecipeMismatch, match="leads_min"):
        render_shards.main(["--zarr", str(synthetic_store), "--out", str(out),
                            "--split", "all", "--leads", "30,60",
                            "--samples-per-shard", "5"])


def test_force_rerender_discards_a_mismatched_store(synthetic_store, tmp_path):
    out = tmp_path / "shards"
    _render(synthetic_store, out)
    assert render_shards.main(["--zarr", str(synthetic_store), "--out", str(out),
                               "--split", "all", "--leads", "30,60",
                               "--samples-per-shard", "5", "--force"]) == 0
    sds = ShardDataset(out / "all")
    assert sds.recipe["leads_min"] == [30, 60]
    assert len(sds) == len(ZarrCorrectionDataset(synthetic_store, leads_min=(30, 60)).index)


def test_expected_channels_mismatch_raises(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    with pytest.raises(ShardRecipeMismatch, match="channels"):
        ShardDataset(shard_dir, expected_channels=999)


def test_expected_channels_env_mismatch_raises(synthetic_store, tmp_path, monkeypatch):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    monkeypatch.setenv("PLUVIO_EXPECTED_CHANNELS", "999")
    with pytest.raises(ShardRecipeMismatch, match="channels"):
        ShardDataset(shard_dir)


def test_edited_manifest_recipe_is_rejected(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    path = shard_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["recipe"]["history_steps"] = 3      # recipe_hash no longer matches
    path.write_text(json.dumps(manifest))
    with pytest.raises(ShardRecipeMismatch, match="recipe_hash"):
        ShardDataset(shard_dir)


def test_not_a_shard_store_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="not a shard store"):
        ShardDataset(tmp_path / "empty")


# ──────────────────────────────────────────────────────────── CLI validation


def test_bad_split_is_rejected(synthetic_store, tmp_path):
    with pytest.raises(SystemExit, match="--split"):
        render_shards.main(["--zarr", str(synthetic_store), "--out", str(tmp_path / "s"),
                            "--split", "test"])


def test_split_all_cannot_be_combined(synthetic_store, tmp_path):
    with pytest.raises(SystemExit, match="do not combine"):
        render_shards.main(["--zarr", str(synthetic_store), "--out", str(tmp_path / "s"),
                            "--split", "all,train"])


# ─────────────────────────────────────────────────────────── train.py wiring


def test_train_swaps_in_the_shard_dataset(synthetic_store, tmp_path):
    from model.train import main as train_main

    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train,val")
    ckpt = tmp_path / "ck" / "unet.pt"
    assert train_main([
        "--shards", str(root), "--zarr", str(synthetic_store),
        "--epochs", "1", "--batch-size", "2", "--base-channels", "4",
        "--num-workers", "0", "--device", "cpu", "--checkpoint", str(ckpt),
    ]) == 0
    state = torch.load(ckpt, weights_only=False)
    assert state["in_channels"] == _zarr_set(synthetic_store).n_channels


def test_train_rejects_shards_rendered_from_a_different_recipe(synthetic_store, tmp_path):
    from model.train import main as train_main

    root = tmp_path / "shards"
    render_shards.main(["--zarr", str(synthetic_store), "--out", str(root),
                        "--split", "train,val", "--leads", "30", "--samples-per-shard", "5"])
    with pytest.raises(ShardRecipeMismatch, match="leads_min"):
        train_main(["--shards", str(root), "--zarr", str(synthetic_store),
                    "--epochs", "1", "--device", "cpu"])


def test_train_rejects_a_rain_filter_the_shards_were_not_rendered_with(synthetic_store, tmp_path):
    from model.train import main as train_main

    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train,val")
    with pytest.raises(SystemExit, match="require-rain-fraction"):
        train_main(["--shards", str(root), "--require-rain-fraction", "0.05",
                    "--epochs", "1", "--device", "cpu"])


def test_train_requires_both_splits(synthetic_store, tmp_path):
    from model.train import main as train_main

    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train")
    with pytest.raises(SystemExit, match="missing"):
        train_main(["--shards", str(root), "--epochs", "1", "--device", "cpu"])


def test_rain_filter_is_rendered_into_the_train_split_only(synthetic_store, tmp_path):
    root = tmp_path / "shards"
    render_shards.main(["--zarr", str(synthetic_store), "--out", str(root),
                        "--split", "train,val", "--leads", LEADS,
                        "--samples-per-shard", "5", "--require-rain-fraction", "0.05"])
    assert ShardDataset(root / "train").recipe["require_rain_fraction"] == 0.05
    assert ShardDataset(root / "val").recipe["require_rain_fraction"] is None
    # ...and it selects the same samples the zarr indexer would.
    zds = _zarr_set(synthetic_store, split="train", require_rain_fraction=0.05)
    assert len(ShardDataset(root / "train")) == len(zds)


# ───────────────────────────────────────────────────────────────── throughput


def test_shard_dataset_is_faster_than_the_zarr_dataset(synthetic_store, tmp_path, capsys):
    """Single-worker CPU samples/s, both datasets, same sample set.

    The synthetic store is 24x24 with 2 aux channels, so the absolute numbers
    mean nothing — the per-sample Python/zarr overhead this replaces is what
    scales with channel count and grid. The real-store number is the acceptance
    gate (>=3x epoch speedup) and needs the GPU box; see docs/training_run_v2.md.
    """
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    zds = _zarr_set(synthetic_store)
    sds = ShardDataset(shard_dir)
    n = len(zds)

    def rate(ds):
        for i in range(min(8, n)):          # warm caches on both sides
            ds[i]
        best = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            for i in range(n):
                ds[i]
            best = max(best, n / (time.perf_counter() - t0))
        return best

    zarr_rate, shard_rate = rate(zds), rate(sds)
    with capsys.disabled():
        print(f"\n  samples/s (CPU, 1 worker, {n} samples, {sds.n_channels}ch @ "
              f"{sds.grid_hw[0]}x{sds.grid_hw[1]}): "
              f"zarr={zarr_rate:.0f} shard={shard_rate:.0f} "
              f"speedup={shard_rate / zarr_rate:.1f}x")
    assert shard_rate > zarr_rate


# ──────────────────────────────────── Lagrangian channels in shards (2.3/2.6)


def test_lagrangian_planes_are_rendered_into_the_shards(synthetic_store, tmp_path):
    """The flow estimate is the expensive half of a Lagrangian sample (2.3), so
    baking it into the shards is the whole point: the rendered sample must be
    the zarr sample, planes and all."""
    shard_dir = _render(synthetic_store, tmp_path / "shards",
                        extra=("--lagrangian-channels", "2"))
    zds = _zarr_set(synthetic_store, lagrangian_channels=2)
    sds = ShardDataset(shard_dir)

    assert sds.n_channels == zds.n_channels
    assert sds.recipe["lagrangian_channels"] == 2
    manifest = json.loads((shard_dir / MANIFEST_NAME).read_text())
    assert manifest["channel_recipe"] == zds.channel_recipe()
    assert manifest["channel_recipe"]["channel_names"][-2:] == [
        "lagrangian_rate", "lagrangian_flow_mag"]

    assert len(sds) == len(zds) > 0
    for i in (0, len(sds) // 2, len(sds) - 1):
        zx, _ = zds[i]
        sx, _ = sds[i]
        assert torch.equal(sx, torch.from_numpy(
            cast_for_shard(zx.numpy(), "float16").astype("float32")))
        assert float(sx[-2].abs().max()) > 0.0      # the advected plane is not empty


def test_recipe_separates_shards_rendered_with_and_without_the_planes(synthetic_store, tmp_path):
    """The channel count alone does not identify a sample set — the recipe must
    name the Lagrangian count, or shards rendered with an extra aux channel and
    no planes would look interchangeable with these."""
    plain = _zarr_set(synthetic_store)
    lag = _zarr_set(synthetic_store, lagrangian_channels=1)
    common = {"split_boundary_epoch": None, "val_frac": None, "dtype": "float16"}
    r_plain = recipe_from_dataset(plain, split="all", **common)
    r_lag = recipe_from_dataset(lag, split="all", **common)
    assert r_plain["lagrangian_channels"] == 0
    assert r_lag["lagrangian_channels"] == 1
    assert any(d.startswith("lagrangian_channels")
               for d in compare_recipes(r_plain, r_lag))

    shard_dir = _render(synthetic_store, tmp_path / "shards")
    with pytest.raises(ShardRecipeMismatch, match="lagrangian_channels"):
        ShardDataset(shard_dir, expected_recipe=r_lag)


def test_train_rejects_shards_missing_the_lagrangian_planes(synthetic_store, tmp_path):
    """`--shards --lagrangian-channels 2` against shards rendered without them
    must fail loudly, not train a model that never sees the planes."""
    from model.train import main as train_main

    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train,val")
    with pytest.raises(SystemExit, match="lagrangian-channels"):
        train_main(["--shards", str(root), "--lagrangian-channels", "2",
                    "--epochs", "1", "--device", "cpu"])


def test_train_probe_compares_the_lagrangian_count_against_the_store(synthetic_store, tmp_path):
    """With --zarr given, the store-derived recipe catches it first, and names
    the field."""
    from model.train import main as train_main

    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train,val")
    with pytest.raises(ShardRecipeMismatch, match="lagrangian_channels"):
        train_main(["--shards", str(root), "--zarr", str(synthetic_store),
                    "--lagrangian-channels", "1", "--epochs", "1", "--device", "cpu"])


def test_train_runs_on_lagrangian_shards_and_records_the_recipe(synthetic_store, tmp_path):
    from model.train import main as train_main

    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train,val",
            extra=("--lagrangian-channels", "1"))
    ckpt = tmp_path / "ck" / "unet.pt"
    assert train_main([
        "--shards", str(root), "--zarr", str(synthetic_store),
        "--lagrangian-channels", "1", "--epochs", "1", "--batch-size", "2",
        "--base-channels", "4", "--num-workers", "0", "--device", "cpu",
        "--checkpoint", str(ckpt),
    ]) == 0
    state = torch.load(ckpt, weights_only=False)
    zds = _zarr_set(synthetic_store, lagrangian_channels=1)
    assert state["in_channels"] == zds.n_channels
    # A shard-trained checkpoint must serve through the same recipe path as a
    # zarr-trained one, or infer_latest would rebuild the wrong input.
    assert state["channel_recipe"]["lagrangian_channels"] == 1
    assert state["channel_recipe"]["n_channels"] == zds.n_channels
    assert state["channel_recipe"]["aux_channels"] == list(zds.aux_channels)
