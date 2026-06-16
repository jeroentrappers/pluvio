# MTG-FCI L2 gridded-product collector — satpy reads the full-disk netCDFs and
# resamples onto our domain, so the full geo stack is needed. Build context =
# research/collectors.
FROM python:3.13
RUN pip install --no-cache-dir eumdac satpy pyresample netcdf4 h5netcdf "rasterio==1.4.*" numpy
COPY fetch_mtg_l2.py /app/fetch_mtg_l2.py
WORKDIR /app
ENTRYPOINT ["python", "fetch_mtg_l2.py"]
