# DWD ICON-D2 NWP collector — downloads regular-lat-lon tot_prec GRIB2 via
# stdlib urllib+bz2 (decoding/regridding happens at zarr-build time), so a bare
# slim Python base is enough. Build context = research/collectors.
FROM python:3.13-slim
COPY fetch_icon_d2.py /app/fetch_icon_d2.py
WORKDIR /app
ENTRYPOINT ["python", "fetch_icon_d2.py"]
