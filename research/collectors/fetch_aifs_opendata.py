"""ECMWF AIFS open-data total precipitation — the medium-range (to 10-day)
NWP context for the seamless model.

AIFS (`aifs-single`) is ECMWF's operational AI forecast, published free on the
open-data portal (CC-BY-4.0, no API key) at 0.25° global, runs 00/06/12/18z,
total precipitation `tp` at 6-hourly steps out to 360 h. We pull the latest run
to a configurable horizon and keep the GRIB2 fields.

`tp` is **accumulated from step 0** (kg m⁻² = mm). To get a per-interval rate
the build step differences consecutive steps and divides by the step hours —
done downstream (build_zarr / build_outlook), not here.

Why we keep the *global* field rather than a Belgium crop: the seamless model's
input context should be far wider than its verified (radar) domain — upstream
synoptic flow drives multi-day skill. The regrid step decides the context
window; the collector just lands the data.

Output: one GRIB2 per step, `aifs_tp_<YYYYmmddTHHZ>_+<step>h.grib2`, under --out.

    python collectors/fetch_aifs_opendata.py --max-step 240 --step 6 --out data/aifs/
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

LOG = logging.getLogger("pluvio.fetch_aifs")


def _members(fc_type: str, n_members: int) -> list[tuple[str, dict]]:
    """The (type, extra-kwargs) list to fetch. Deterministic = one `fc`;
    ensemble = control `cf` + `n_members` perturbed `pf` (gives the spread =
    free medium-range uncertainty)."""
    if n_members <= 0:
        return [(fc_type, {})]
    return [("cf", {})] + [("pf", {"number": m}) for m in range(1, n_members + 1)]


def fetch_run(
    out_dir: pathlib.Path,
    model: str,
    param: str,
    fc_type: str,
    steps: list[int],
    n_members: int,
) -> tuple[int, str | None]:
    """Download `param` at each step (and ensemble member) of the latest run."""
    from ecmwf.opendata import Client

    client = Client(source="ecmwf", model=model)
    members = _members(fc_type, n_members)

    # Resolve the latest available run once (so every step comes from the same
    # forecast), then pull each step × member into its own file.
    latest = client.latest(type=members[0][0], param=param, step=steps[0], **members[0][1])
    run_id = latest.strftime("%Y%m%dT%HZ")
    LOG.info("latest %s run: %s — %d steps × %d members (%s…%sh)",
             model, run_id, len(steps), len(members), steps[0], steps[-1])

    n_written = 0
    for step in steps:
        for typ, extra in members:
            tag = typ if not extra else f"{typ}{extra['number']:02d}"
            target = out_dir / f"{model}_{param}_{run_id}_+{step}h_{tag}.grib2"
            if target.exists() and target.stat().st_size > 0:
                n_written += 1
                continue
            try:
                client.retrieve(date=latest, type=typ, param=param, step=step,
                                target=str(target), **extra)
                n_written += 1
            except Exception as exc:  # noqa: BLE001 — one bad fetch shouldn't kill the run
                LOG.warning("step +%dh %s failed (%s); skipping", step, tag, exc)
                target.unlink(missing_ok=True)
    return n_written, run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="aifs-single", help="open-data model id (aifs-single | aifs-ens)")
    parser.add_argument("--param", default="tp", help="GRIB short name (total precipitation)")
    parser.add_argument("--type", dest="fc_type", default="fc", help="deterministic type (fc); ignored when --members>0")
    parser.add_argument("--members", type=int, default=0,
                        help="ensemble: control + N perturbed members (use with --model aifs-ens)")
    parser.add_argument("--max-step", type=int, default=240, help="forecast horizon, hours (≤360)")
    parser.add_argument("--step", type=int, default=6, help="step cadence, hours")
    parser.add_argument("--out", default="data/aifs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    steps = list(range(args.step, args.max_step + 1, args.step))
    if not steps:
        LOG.error("no steps to fetch (check --step/--max-step)")
        return 2

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n, run_id = fetch_run(out_dir, args.model, args.param, args.fc_type, steps, args.members)
    if n == 0:
        LOG.error("no fields fetched")
        return 1
    LOG.info("done: %d fields for run %s in %s", n, run_id, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
