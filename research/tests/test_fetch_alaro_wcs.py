"""The ALARO collector's WCS mode requests physical coverages and refuses
non-image bodies; the WMS path stays a rendered mask."""

from __future__ import annotations

import datetime as dt
import pathlib
from datetime import datetime

import pytest

from collectors import fetch_alaro_24h as fa


class _Resp:
    def __init__(self, content: bytes, ctype: str, status: int = 200):
        self.content, self.headers, self.status_code = content, {"content-type": ctype}, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Client:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return self.resp


def test_wcs_request_targets_the_coverage_with_time_and_bbox_subsets(tmp_path):
    client = _Client(_Resp(b"II*\x00fake-tiff", "image/tiff"))
    when = datetime(2026, 9, 4, 12, 0, tzinfo=dt.UTC)
    out = tmp_path / "alaro_TPmm_20260904T120000Z.tif"
    fa.fetch_wcs_coverage(client, "Total_precipitation", when, (0.0, 48.5, 11.0, 56.0), out)
    url, params = client.calls[0]
    assert url == fa.WCS_URL
    p = {k: v for k, v in params if k != "subset"}
    subsets = [v for k, v in params if k == "subset"]
    assert p["request"] == "GetCoverage" and p["coverageId"] == "alaro__Total_precipitation"
    assert "Lat(48.5,56.0)" in subsets and "Long(0.0,11.0)" in subsets
    assert 'time("2026-09-04T12:00:00.000Z")' in subsets
    assert out.read_bytes().startswith(b"II*")


def test_wcs_refuses_a_service_exception_body(tmp_path):
    body = b'<?xml version="1.0"?><ows:ExceptionReport><ServiceException>no such time</ServiceException></ows:ExceptionReport>'
    client = _Client(_Resp(body, "application/xml"))
    out = tmp_path / "x.tif"
    with pytest.raises(fa.WMSError):
        fa.fetch_wcs_coverage(client, "Total_precipitation", datetime(2026, 9, 4, tzinfo=dt.UTC),
                              (0.0, 48.5, 11.0, 56.0), out)
    assert not out.exists()


def test_cli_rejects_wcs_for_unmapped_layers(monkeypatch):
    assert fa.main(["--layer", "Surface_CAPE", "--wcs", "--out", "/tmp/x"]) == 2


def test_build_zarr_registers_the_physical_channel_from_the_wcs_filenames():
    from tools import build_zarr as bz

    ch = {c.var: c for c in bz.ALARO_CHANNELS}
    assert "alaro_precip_mm" in ch
    assert ch["alaro_precip_mm"].layer == fa.WCS_FILE_LAYER["Total_precipitation"] == "TPmm"
    d, pattern = ch["alaro_precip_mm"].dir_and_pattern(pathlib.Path("/data"))
    assert d == pathlib.Path("/data/alaro") and pattern == "alaro_TPmm_*.tif"
