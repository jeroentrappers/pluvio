# Slim CPU image to build the seamless zarr on hetz1 (no GPU/torch needed — the
# build is reprojection only). Build context must contain model/ and tools/.
FROM python:3.13
RUN pip install --no-cache-dir "rasterio==1.4.*" pyproj zarr numpy
COPY model /app/model
COPY tools /app/tools
WORKDIR /app
ENTRYPOINT ["python", "tools/build_seamless_zarr.py"]
