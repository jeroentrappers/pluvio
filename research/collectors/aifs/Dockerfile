# AIFS NWP context collector — just downloads GRIB via ecmwf-opendata (decoding
# /regridding happens at zarr-build time), so a slim Python base with no GDAL is
# enough. Build context = research/collectors.
FROM python:3.13-slim
RUN pip install --no-cache-dir ecmwf-opendata httpx
COPY fetch_aifs_opendata.py /app/fetch_aifs_opendata.py
WORKDIR /app
ENTRYPOINT ["python", "fetch_aifs_opendata.py"]
