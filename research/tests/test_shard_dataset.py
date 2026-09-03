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
    ShardSourceMismatch,
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
    # Guards the test above from being vacuous: if every channel happened to be
    # float16-exact, the two equality tests would be the same test. Most of them
    # ARE exact — `_normalise` divides float16 store arrays by python floats, so
    # the arithmetic stays float16, and `lead/120` is exact for every lead the
    # 30-min cadence allows. The tod sin/cos planes are what quantise (≤2.4e-4).
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


def test_bumped_normalise_version_is_refused_without_a_store_to_compare_against(
        synthetic_store, tmp_path, monkeypatch):
    """`--shards` without `--zarr` has no store to re-derive a recipe from, but
    NORMALISE_VERSION is a constant of this build — so the "stale shards are
    refused after _normalise changes" guarantee must not depend on --zarr."""
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    monkeypatch.setattr("model.zarr_dataset.NORMALISE_VERSION", 99)
    with pytest.raises(ShardRecipeMismatch, match="normalise_version"):
        ShardDataset(shard_dir)


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


# ────────────────────────────────── source-store identity on resume (2.6 F1)


def _rebuild_store_values(store, *, identical: bool, seed: int = 7) -> None:
    """Rewrite the store's per-issue arrays IN PLACE, leaving the structure
    (array inventory, shapes, dtypes, attrs) and the whole ``issue_time``
    vector untouched — i.e. exactly what a re-run of the store builder over the
    same window produces. ``identical=True`` writes the same bytes back, so the
    two cases differ only in the VALUES."""
    import zarr

    root = zarr.open_group(str(store), mode="a")
    rng = np.random.default_rng(seed)
    for name in sorted(root.array_keys()):
        if name in ("issue_time", "leads_min"):
            continue
        arr = np.asarray(root[name][:])
        root[name][:] = arr if identical else (
            rng.random(arr.shape) * 5.0).astype(arr.dtype)


def test_source_store_hash_catches_an_in_place_value_rebuild(synthetic_store):
    """The failure the structural-only hash missed: same arrays, same shapes,
    same attrs, same issue_time, different numbers."""
    before = render_shards.source_store_hash(synthetic_store)
    assert before["hash_mode"] == "structural+sampled"
    assert before["sampled"]["n_issues_sampled"] > 0
    assert "radar[:, 0]" in before["sampled"]["arrays"]

    _rebuild_store_values(synthetic_store, identical=True)
    assert render_shards.source_store_hash(synthetic_store)["hash"] == before["hash"]

    _rebuild_store_values(synthetic_store, identical=False)
    assert render_shards.source_store_hash(synthetic_store)["hash"] != before["hash"]


def test_resume_refuses_a_source_store_rebuilt_in_place(synthetic_store, tmp_path):
    """Without this, the resume re-rendered ONLY the missing shards from the new
    store and ShardDataset accepted a directory that is half one store and half
    another (index.npy silently overwritten from the new index on top)."""
    out = tmp_path / "shards"
    shard_dir = _render(synthetic_store, out)
    (shard_dir / "x_00001.npy").unlink()

    _rebuild_store_values(synthetic_store, identical=False)
    with pytest.raises(ShardSourceMismatch, match="--force"):
        _render(synthetic_store, out)
    # ...and nothing was rendered into the hole in the meantime.
    assert not (shard_dir / "x_00001.npy").exists()


def test_resume_proceeds_when_the_rebuild_left_the_values_identical(synthetic_store, tmp_path):
    """The check must be a fingerprint of the CONTENT, not of the mtime: a
    rebuild that reproduces the same numbers is the same store, and re-rendering
    142k samples over it would be a day of CPU for nothing."""
    out = tmp_path / "shards"
    shard_dir = _render(synthetic_store, out)
    reference = json.loads((shard_dir / MANIFEST_NAME).read_text())
    (shard_dir / "x_00001.npy").unlink()

    _rebuild_store_values(synthetic_store, identical=True)
    _render(synthetic_store, out)
    resumed = json.loads((shard_dir / MANIFEST_NAME).read_text())
    assert resumed["complete"] is True
    assert [(s["id"], s["sha256_x"]) for s in resumed["shards"]] == \
           [(s["id"], s["sha256_x"]) for s in reference["shards"]]
    ShardDataset(shard_dir, verify_checksums=True)


def test_force_re_renders_after_a_source_store_rebuild(synthetic_store, tmp_path):
    out = tmp_path / "shards"
    _render(synthetic_store, out)
    _rebuild_store_values(synthetic_store, identical=False)
    _render(synthetic_store, out, extra=("--force",))
    sds = ShardDataset(out / "all", verify_checksums=True)
    assert sds.manifest["source_store"]["hash"] == \
        render_shards.source_store_hash(synthetic_store)["hash"]


def test_train_refuses_shards_rendered_from_a_since_rebuilt_store(synthetic_store, tmp_path):
    from model.train import main as train_main

    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train,val")
    _rebuild_store_values(synthetic_store, identical=False)
    with pytest.raises(ShardSourceMismatch, match="fingerprint"):
        train_main(["--shards", str(root), "--zarr", str(synthetic_store),
                    "--epochs", "1", "--device", "cpu"])


# ────────────────────────────────────────── per-issue dedup layout (2.6 F2)


def test_dedup_is_the_default_layout_and_stores_one_block_per_issue(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    manifest = json.loads((shard_dir / MANIFEST_NAME).read_text())
    zds = _zarr_set(synthetic_store)

    assert manifest["layout"] == "dedup"
    assert manifest["lead_varying_channels"] == list(zds.lead_varying_channels())
    meta = np.load(shard_dir / INDEX_NAME)
    for shard in manifest["shards"]:
        lo = shard["first_sample"]
        block = meta["issue_idx"][lo:lo + shard["n_samples"]]
        assert shard["n_issues"] == len({int(v) for v in block})
        inv = np.load(shard_dir / shard["inv"], mmap_mode="r")
        assert inv.shape == (shard["n_issues"],
                             zds.n_channels - len(zds.lead_varying_channels()),
                             *zds.grid_hw)
        assert np.load(shard_dir / shard["x"], mmap_mode="r").shape == (
            shard["n_samples"], len(zds.lead_varying_channels()), *zds.grid_hw)
        assert len(shard["sha256_inv"]) == 64


def test_flat_and_dedup_layouts_hand_out_the_same_samples_and_dedup_is_smaller(
        synthetic_store, tmp_path):
    """The dedup layout is a storage change only: the reassembled sample must be
    the flat sample bit for bit, at measurably less disk."""
    dedup = _render(synthetic_store, tmp_path / "dedup")
    flat = _render(synthetic_store, tmp_path / "flat", extra=("--layout", "flat"))
    d, f = ShardDataset(dedup), ShardDataset(flat)
    assert (d.layout, f.layout) == ("dedup", "flat")
    assert len(d) == len(f) > 0
    for i in range(len(f)):
        dx, dy = d[i]
        fx, fy = f[i]
        assert torch.equal(dx, fx), f"input mismatch at sample {i}"
        assert torch.equal(dy, fy), f"target mismatch at sample {i}"

    def npy_bytes(p):
        return sum(q.stat().st_size for q in p.iterdir() if q.suffix == ".npy")

    assert npy_bytes(dedup) < npy_bytes(flat)
    # The two files are the layouts' only difference, so no `inv` in flat.
    assert not any(q.name.startswith("inv_") for q in flat.iterdir())


def test_dedup_layout_is_equally_exact_with_the_lagrangian_planes(synthetic_store, tmp_path):
    """`lagrangian_flow_mag` is lead-INdependent and `lagrangian_rate` is not,
    so the two Lagrangian planes land on opposite sides of the split — the case
    most likely to be got wrong."""
    shard_dir = _render(synthetic_store, tmp_path / "shards",
                        extra=("--lagrangian-channels", "2"))
    zds = _zarr_set(synthetic_store, lagrangian_channels=2)
    sds = ShardDataset(shard_dir)
    names = json.loads((shard_dir / MANIFEST_NAME).read_text())["channel_recipe"]["channel_names"]
    varying = [names[i] for i in sds._var_idx]
    assert "lagrangian_rate" in varying and "lagrangian_flow_mag" not in varying
    assert len(sds) == len(zds) > 0
    for i in range(len(zds)):
        zx, zy = zds[i]
        sx, sy = sds[i]
        assert torch.equal(sx, torch.from_numpy(
            cast_for_shard(zx.numpy(), "float16").astype("float32"))), i
        assert torch.equal(sy, torch.from_numpy(
            cast_for_shard(zy.numpy(), "float16").astype("float32"))), i


def test_the_declared_lead_varying_channels_are_exactly_the_ones_that_vary(synthetic_store):
    """Measured against ``build_input`` itself rather than asserted from the
    channel names: every channel the dedup layout stores once per issue really
    is identical across that issue's leads, and every channel it stores per
    sample really does change."""
    zds = _zarr_set(synthetic_store, lagrangian_channels=2)
    var = set(zds.lead_varying_channels())
    inv = [c for c in range(zds.n_channels) if c not in var]

    by_issue: dict[int, list[int]] = {}
    for i, s in enumerate(zds.index):
        by_issue.setdefault(s.issue_idx, []).append(i)
    groups = [g for g in by_issue.values() if len(g) > 1]
    assert groups, "test store must have an issue with several leads"

    seen_varying: set[int] = set()
    for group in groups:
        x0 = zds[group[0]][0].numpy()
        for j in group[1:]:
            xj = zds[j][0].numpy()
            for c in inv:
                assert np.array_equal(x0[c], xj[c]), f"channel {c} is not lead-invariant"
            seen_varying |= {c for c in var if not np.array_equal(x0[c], xj[c])}
    assert seen_varying == var, (
        f"channels {sorted(var - seen_varying)} are declared lead-varying but never "
        "changed between an issue's leads — they belong in the per-issue block")


def test_dedup_render_refuses_a_channel_that_is_not_actually_lead_invariant(
        synthetic_store, tmp_path, monkeypatch):
    """The renderer verifies the split per issue, so drifting
    LEAD_VARYING_CHANNEL_NAMES out of step with build_input is a loud render
    failure and not 113k silently-wrong samples."""
    import model.zarr_dataset as zd

    monkeypatch.setattr(
        zd, "LEAD_VARYING_CHANNEL_NAMES",
        zd.LEAD_VARYING_CHANNEL_NAMES - {"nowcast_at_lead"})
    with pytest.raises(RuntimeError, match="nowcast_at_lead"):
        _render(synthetic_store, tmp_path / "shards")


def test_layouts_cannot_be_mixed_in_one_directory(synthetic_store, tmp_path):
    out = tmp_path / "shards"
    _render(synthetic_store, out)
    with pytest.raises(ShardRecipeMismatch, match="layout"):
        _render(synthetic_store, out, extra=("--layout", "flat"))


def test_dedup_store_without_channel_names_is_refused(synthetic_store, tmp_path):
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    path = shard_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["channel_recipe"].pop("channel_names")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ShardRecipeMismatch, match="channel_names"):
        ShardDataset(shard_dir)


# ─────────────────────────────────────── index.npy, --force, manifest drift


def test_missing_index_is_refused_rather_than_zero_filled(synthetic_store, tmp_path):
    """index.npy used to be optional, falling back to zeros — which turned a
    lost index into a store where every sample claims issue_epoch 0."""
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    (shard_dir / INDEX_NAME).unlink()
    with pytest.raises(ShardStoreIncomplete, match=INDEX_NAME):
        ShardDataset(shard_dir)


def test_force_prunes_shard_files_the_new_plan_does_not_name(synthetic_store, tmp_path):
    """A --force re-render of a SMALLER sample set used to leave the previous
    run's higher-numbered shards on disk — unreferenced, so invisible to the
    loader, and tens of GiB of it on the real store."""
    out = tmp_path / "shards"
    _render(synthetic_store, out)
    shard_dir = out / "all"
    before = {p.name for p in shard_dir.iterdir() if p.suffix == ".npy"}
    (shard_dir / "orphan_00099.npy").write_bytes(b"junk")
    (shard_dir / "x_00099.npy.tmp").write_bytes(b"junk")

    assert render_shards.main(["--zarr", str(synthetic_store), "--out", str(out),
                               "--split", "all", "--leads", "30",
                               "--samples-per-shard", "5", "--force"]) == 0
    manifest = json.loads((shard_dir / MANIFEST_NAME).read_text())
    named = {INDEX_NAME}
    for shard in manifest["shards"]:
        named |= {shard["x"], shard["y"], shard["inv"]}
    left = {p.name for p in shard_dir.iterdir()
            if p.suffix == ".npy" or p.name.endswith(".npy.tmp")}
    assert left == named, f"stale files survived --force: {sorted(left - named)}"
    assert len(before) > len(named), "test needs the second render to plan fewer shards"


def test_a_pre_2_3_manifest_error_names_the_missing_recipe_keys(synthetic_store, tmp_path):
    """A store rendered before `lagrangian_channels` joined RECIPE_KEYS fails
    the self-hash check; the message has to say WHY, or it reads as a
    hand-edited manifest."""
    shard_dir = _render(synthetic_store, tmp_path / "shards")
    path = shard_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["recipe"].pop("lagrangian_channels")
    manifest["recipe"]["some_removed_key"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ShardRecipeMismatch) as excinfo:
        ShardDataset(shard_dir)
    msg = str(excinfo.value)
    assert "recipe_hash" in msg
    assert "lagrangian_channels" in msg and "MISSING" in msg
    assert "some_removed_key" in msg


# ───────────────────────────────── checkpoint provenance / recipe (2.6, 2.3)


def test_shard_and_zarr_trained_checkpoints_carry_the_same_channel_recipe(
        synthetic_store, tmp_path):
    """2.3 left train.py hand-building a subset of the recipe on the shard path
    (no channel_names), so the two paths produced different checkpoints from the
    same channel layout."""
    from model.train import main as train_main

    root = tmp_path / "shards"
    _render(synthetic_store, root, split="train,val", extra=("--lagrangian-channels", "1"))
    common = ["--epochs", "1", "--batch-size", "2", "--base-channels", "4",
              "--num-workers", "0", "--device", "cpu", "--lagrangian-channels", "1"]
    shard_ckpt = tmp_path / "ck" / "shard.pt"
    zarr_ckpt = tmp_path / "ck" / "zarr.pt"
    assert train_main(["--shards", str(root), "--zarr", str(synthetic_store),
                       "--checkpoint", str(shard_ckpt), *common]) == 0
    assert train_main(["--zarr", str(synthetic_store),
                       "--checkpoint", str(zarr_ckpt), *common]) == 0

    from_shards = torch.load(shard_ckpt, weights_only=False)["channel_recipe"]
    from_zarr = torch.load(zarr_ckpt, weights_only=False)["channel_recipe"]
    assert from_shards == from_zarr
    assert from_shards["channel_names"]


def test_checkpoint_records_which_shard_store_it_trained_on(synthetic_store, tmp_path):
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
    prov = state["shards"]
    manifest = json.loads((root / "train" / MANIFEST_NAME).read_text())
    assert prov["root"] == str(root)
    assert prov["recipe_hash"] == manifest["recipe_hash"]
    assert prov["source_store"]["hash"] == \
        render_shards.source_store_hash(synthetic_store)["hash"]


def test_a_zarr_trained_checkpoint_has_no_shard_provenance(synthetic_store, tmp_path):
    from model.train import main as train_main

    ckpt = tmp_path / "ck" / "unet.pt"
    assert train_main(["--zarr", str(synthetic_store), "--epochs", "1", "--batch-size", "2",
                       "--base-channels", "4", "--num-workers", "0", "--device", "cpu",
                       "--checkpoint", str(ckpt)]) == 0
    assert torch.load(ckpt, weights_only=False)["shards"] is None
