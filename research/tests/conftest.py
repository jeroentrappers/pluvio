"""Test config: make ``model`` (and friends under research/) importable."""

from __future__ import annotations

import pathlib
import sys

_RESEARCH_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))
