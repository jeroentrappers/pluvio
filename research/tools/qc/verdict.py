"""One JSON verdict schema for QC checks: {generated, checks: [...], summary}.

Each check is {name, status: ok|warn|crit, value, threshold, detail}. Callers
build a list of `Check` and pass it to `build_verdict`; `to_json` /
`write_atomic` handle numpy scalar/array serialisation so no check function
has to remember to call `.item()` before returning a value.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from typing import Any

STATUSES = ("ok", "warn", "crit")


@dataclasses.dataclass
class Check:
    name: str
    status: str  # "ok" | "warn" | "crit"
    value: Any = None
    threshold: Any = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Check {self.name!r}: bad status {self.status!r}")


def _default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"object of type {type(o)!r} is not JSON serialisable")


def worst_status(checks: list[Check]) -> str:
    statuses = {c.status for c in checks}
    if "crit" in statuses:
        return "crit"
    if "warn" in statuses:
        return "warn"
    return "ok"


def build_verdict(checks: list[Check], generated: str | None = None) -> dict:
    """Assemble the one-schema verdict dict from a list of Check results."""
    generated = generated or dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return {
        "generated": generated,
        "checks": [dataclasses.asdict(c) for c in checks],
        "summary": worst_status(checks),
    }


def to_json(verdict: dict, *, indent: int = 1) -> str:
    return json.dumps(verdict, indent=indent, default=_default)


def write_atomic(path: str | pathlib.Path, body: dict, *, indent: int = 1) -> pathlib.Path:
    """Write `body` (any JSON-able dict, numpy scalars included) atomically."""
    op = pathlib.Path(path)
    tmp = op.with_name(op.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=indent, default=_default))
    tmp.replace(op)
    return op


def exit_code(verdict_or_checks) -> int:
    """1 if any check (or the verdict's summary) is warn/crit, else 0."""
    if isinstance(verdict_or_checks, dict):
        return 0 if verdict_or_checks.get("summary", "ok") == "ok" else 1
    return 0 if worst_status(verdict_or_checks) == "ok" else 1
