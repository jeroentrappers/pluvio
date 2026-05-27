"""md5 signing for the KMI mobile-app API.

Mirrors ``lib/features/radar/data/api_signing.dart`` and the Python
collector under ``research/collectors/_kmi_signing.py``. The salt is the
one shared with the upstream Apache-2.0 ``irm-kmi-api`` package; rotates
daily on device-local date.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

_SALT = "r9EnW374jkJ9acc"


def sign(method: str, now: datetime | None = None) -> str:
    when = now or datetime.now()
    payload = f"{_SALT};{method};{when:%d/%m/%Y}".encode()
    return hashlib.md5(payload).hexdigest()
