# MTG Lightning Imager collector — fetches LI L2 accumulated products via EUMDAC,
# projects GEOS→lat/lon and grids them. Needs the full Python image (rasterio +
# netCDF + pyproj). Build context = research/collectors.
FROM python:3.13
RUN pip install --no-cache-dir eumdac xarray h5netcdf netcdf4 pyproj "rasterio==1.4.*" numpy
COPY fetch_mtg_lightning.py /app/fetch_mtg_lightning.py
WORKDIR /app
ENTRYPOINT ["python", "fetch_mtg_lightning.py"]
