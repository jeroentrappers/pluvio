"""Reimplementation of the KMI mobile-app daily-rotating md5 signature.

Mirrors `lib/features/radar/data/api_signing.dart` so the Python collector
hits the same endpoints the Flutter app does. The salt is the one used
upstream by the Apache-2.0 `irm-kmi-api` package; if KMI rotates it, both
implementations break together and both need updating.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

_SALT = "r9EnW374jkJ9acc"


def sign(method: str, now: datetime | None = None) -> str:
    """Return the `k=` query value the KMI app API expects for ``method``."""
    when = now or datetime.now()
    payload = f"{_SALT};{method};{when:%d/%m/%Y}".encode()
    return hashlib.md5(payload).hexdigest()  # noqa: S324  — same algorithm as upstream
