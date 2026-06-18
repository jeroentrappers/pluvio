# Open-Meteo high-res NWP collector — queries the Historical Forecast API and
# writes ERA5-shaped monthly NetCDFs (consumed by build_pretrain_zarr --openmeteo).
# Build context = research/collectors.
FROM python:3.13-slim
RUN pip install --no-cache-dir requests xarray netcdf4 numpy
COPY fetch_openmeteo_nwp.py /app/fetch_openmeteo_nwp.py
WORKDIR /app
ENTRYPOINT ["python", "fetch_openmeteo_nwp.py"]
