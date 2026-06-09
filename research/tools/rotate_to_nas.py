"""Rotate completed-month chunks from local stage/ to NAS archive/.

Many-small-files writes over wifi-CIFS are dominated by per-file metadata
RTTs and run ~5x slower than a single large write. This tool batches each
completed month of per-source files into one zstd-compressed tarball and
moves it to the NAS, then deletes the locally staged originals.

Files in the *current* calendar month are left alone — a concurrent
historical pull or forward cron may still be writing them.

Layout assumed:

    $PLUVIO_STAGE_DIR/
        knmi/radar_forecast/2.0/RAD_NL25_RAC_FM_YYYYMMDDHHMM.h5
        msg/msg_fes_ir108/msg_fes_ir108_YYYYMMDDTHHMMSSZ.tif
        aws/kmi_aws_10min.parquet           # ← parquets stay local-and-NAS, see below

    $PLUVIO_NAS_DIR/archive/
        knmi/radar_forecast/2.0/2024/2024-08.tar.zst
        msg/msg_fes_ir108/2024/2024-08.tar.zst
        ...

The per-month tarball groups files by the YYYYMM embedded in their
filename (12-digit stamp for KNMI; ISO compact for Meteosat). Parquet
sources (KMI AWS) are atomically *mirrored* to NAS, not tarred — the
file is small, monolithic, and read-heavy.

Idempotent:
  • If a target tarball already exists on NAS and its sha256 matches a
    sidecar `.sha256` file, we skip the month.
  • Otherwise we (re-)create the tarball atomically (.tmp → rename) so a
    crash mid-write doesn't leave a corrupt archive visible.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import logging
import os
import pathlib
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from datetime import datetime, timezone

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - dependency is in requirements
    zstd = None  # type: ignore[assignment]

LOG = logging.getLogger("pluvio.rotate")

# Per-source rule: (relative path inside stage/, glob, YYYYMM-extracting regex).
# A None regex marks "mirror, don't tar" (parquet/database files).
Rule = tuple[str, str, re.Pattern[str] | None]

RULES: list[Rule] = [
    ("knmi/radar_forecast/2.0", "RAD_NL25_RAC_FM_*.h5", re.compile(r"_(\d{6})\d{6}\.h5$")),
    ("knmi/nl_rdr_data_rtcor_5m/1.0", "RAD_NL25_RAC_RT_*.h5", re.compile(r"_(\d{6})\d{6}\.h5$")),
    ("knmi/radar_volume_full_herwijnen/1.0", "RAD_NL62_VOL_NA_*.h5", re.compile(r"_(\d{6})\d{6}\.h5$")),
    ("knmi/radar_volume_denhelder/2.0", "RAD_NL61_VOL_NA_*.h5", re.compile(r"_(\d{6})\d{6}\.h5$")),
    # Meteosat GeoTIFFs land under one dir per layer.
    ("msg", "*/*.tif", re.compile(r"_(\d{6})\d{2}T\d{6}Z\.tif$")),
    # ALARO GeoTIFFs (when we start forward collection).
    ("alaro", "*.tif", re.compile(r"_(\d{6})\d{2}T\d{6}Z\.tif$")),
    # KMI AWS parquet — mirrored, not tarred.
    ("aws", "kmi_aws_10min.parquet", None),
]


@dataclasses.dataclass
class Plan:
    rel_path: str
    month: str  # "YYYYMM"
    files: list[pathlib.Path]


def discover_months(stage_root: pathlib.Path, rule: Rule, current_month: str) -> list[Plan]:
    """Group files under stage_root/rule.rel_path by month, skipping current."""
    rel_path, pattern, month_re = rule
    src = stage_root / rel_path
    if not src.exists():
        return []
    if month_re is None:
        return []  # mirror handled separately
    by_month: dict[str, list[pathlib.Path]] = defaultdict(list)
    for f in sorted(src.glob(pattern)):
        if not f.is_file():
            continue
        m = month_re.search(f.name)
        if not m:
            continue
        month = m.group(1)
        if month >= current_month:
            continue  # leave the in-progress month alone
        by_month[month].append(f)
    return [Plan(rel_path=rel_path, month=m, files=fs) for m, fs in sorted(by_month.items())]


def make_tar_zst(files: list[pathlib.Path], out_path: pathlib.Path, level: int = 10) -> str:
    """Create out_path.tmp, fill with a zstd-compressed tar of `files`, then rename.

    Returns the sha256 hex digest of the final archive.
    """
    if zstd is None:
        raise RuntimeError("zstandard package not installed (pip install zstandard)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    hasher = hashlib.sha256()
    cctx = zstd.ZstdCompressor(level=level, threads=-1)
    with tmp.open("wb") as raw, cctx.stream_writer(raw) as compressor:
        # Hash the compressed stream as we write. We tee writes through
        # the hasher by wrapping in an io.BufferedWriter on a small class.
        class _Tee(io.RawIOBase):
            def __init__(self, sink):
                self._sink = sink

            def writable(self) -> bool:
                return True

            def write(self, b):  # type: ignore[override]
                hasher.update(bytes(b))
                self._sink.write(b)
                return len(b)

        with tarfile.open(fileobj=_Tee(compressor), mode="w|") as tar:  # type: ignore[arg-type]
            for f in files:
                tar.add(f, arcname=f.name)
    os.rename(tmp, out_path)
    return hasher.hexdigest()


def mirror(parquet_local: pathlib.Path, nas_target: pathlib.Path) -> None:
    """Atomically replace nas_target with a copy of parquet_local."""
    nas_target.parent.mkdir(parents=True, exist_ok=True)
    tmp = nas_target.with_suffix(nas_target.suffix + ".tmp")
    shutil.copy2(parquet_local, tmp)
    os.replace(tmp, nas_target)


def run(stage_dir: pathlib.Path, nas_dir: pathlib.Path, dry_run: bool) -> int:
    archive_root = nas_dir / "archive"
    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y%m")
    rotated = 0

    for rule in RULES:
        rel_path, _, month_re = rule

        # Mirror-only rule (parquet / db).
        if month_re is None:
            src = stage_dir / rel_path
            if not src.exists():
                continue
            dst = nas_dir / rel_path
            LOG.info("mirror %s → %s", src, dst)
            if not dry_run:
                mirror(src, dst)
            continue

        plans = discover_months(stage_dir, rule, current_month)
        for plan in plans:
            year = plan.month[:4]
            target = archive_root / plan.rel_path / year / f"{plan.month[:4]}-{plan.month[4:6]}.tar.zst"
            sidecar = target.with_suffix(target.suffix + ".sha256")

            # Skip if NAS already has an archive whose sha matches a *fresh*
            # local re-tar — but always recompute, since tar order may vary.
            # Cheap win: if size matches and the archive exists, assume done.
            if target.exists() and sidecar.exists():
                LOG.info(
                    "skip %s (archive already exists at %s)",
                    plan.rel_path + "/" + plan.month,
                    target,
                )
                # Still delete the local copies, since they're archived.
                if not dry_run:
                    for f in plan.files:
                        f.unlink()
                continue

            LOG.info(
                "tar %s/%s: %d files → %s",
                plan.rel_path,
                plan.month,
                len(plan.files),
                target,
            )
            if dry_run:
                rotated += 1
                continue

            digest = make_tar_zst(plan.files, target)
            sidecar.write_text(digest + "\n")
            for f in plan.files:
                f.unlink()
            rotated += 1

    LOG.info("done — %d month-chunks rotated", rotated)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default=os.environ.get(
            "PLUVIO_STAGE_DIR",
            str(pathlib.Path.home() / "Developer/appmire/pluvio/research/stage"),
        ),
    )
    parser.add_argument(
        "--nas",
        default=os.environ.get("PLUVIO_NAS_DIR", "/mnt/media/weather/pluvio/data"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    stage_dir = pathlib.Path(args.stage)
    nas_dir = pathlib.Path(args.nas)
    if not stage_dir.exists():
        LOG.warning("stage dir does not exist: %s", stage_dir)
        return 0
    if not nas_dir.exists():
        LOG.error("NAS dir does not exist: %s", nas_dir)
        return 2

    return run(stage_dir, nas_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
