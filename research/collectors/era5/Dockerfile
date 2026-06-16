# ERA5 reanalysis collector — opens the public ARCO-ERA5 Zarr on GCS
# (anonymous, no credentials) and pulls our regional crop to NetCDF.
# Regridding onto the analysis grid happens at zarr-build time, not here.
# Build context = research/collectors.
FROM python:3.13-slim
RUN pip install --no-cache-dir \
    "xarray" "zarr" "gcsfs" "dask" "netcdf4" "numpy"
COPY fetch_era5.py /app/fetch_era5.py
WORKDIR /app
ENTRYPOINT ["python", "fetch_era5.py"]
