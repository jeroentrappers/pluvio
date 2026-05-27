"""Band definitions and refresh cadences.

Pluvio's forecast is split into four lead-time *bands*. Each band has its own
refresh cadence so we don't waste compute re-running the 240-hour outlook
every 5 minutes. The model that powers each band is also free to differ —
the nowcast band today uses a fast extrapolation stub; the long-range band
will eventually use AIFS.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

BandName = Literal["nowcast", "short", "medium", "long"]


@dataclasses.dataclass(frozen=True)
class Band:
    """Static description of one lead-time band.

    ``lead_min_start`` is inclusive, ``lead_min_end`` exclusive. The set of
    integer lead minutes a band emits is ``range(start, end, step_min)``.
    """

    name: BandName
    lead_min_start: int
    lead_min_end: int
    step_min: int
    refresh_seconds: int
    cron_expression: str  # standard 5-field cron; informational, used by the example crontab

    @property
    def leads_min(self) -> list[int]:
        return list(range(self.lead_min_start, self.lead_min_end, self.step_min))

    @property
    def n_leads(self) -> int:
        return len(self.leads_min)


BANDS: dict[BandName, Band] = {
    "nowcast": Band(
        name="nowcast",
        lead_min_start=0,
        lead_min_end=120,
        step_min=10,
        refresh_seconds=300,
        cron_expression="*/5 * * * *",
    ),
    "short": Band(
        name="short",
        lead_min_start=120,
        lead_min_end=12 * 60,
        step_min=60,
        refresh_seconds=3600,
        cron_expression="0 * * * *",
    ),
    "medium": Band(
        name="medium",
        lead_min_start=12 * 60,
        lead_min_end=24 * 60,
        step_min=60,
        refresh_seconds=3 * 3600,
        cron_expression="0 */3 * * *",
    ),
    "long": Band(
        name="long",
        lead_min_start=24 * 60,
        lead_min_end=240 * 60,
        step_min=3 * 60,
        refresh_seconds=12 * 3600,
        cron_expression="0 0,12 * * *",
    ),
}


def band(name: BandName) -> Band:
    return BANDS[name]


def all_bands() -> list[Band]:
    return list(BANDS.values())
