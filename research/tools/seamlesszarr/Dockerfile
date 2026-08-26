# Slim CPU image to build the seamless zarr on hetz1 (no GPU/torch needed — the
# build is reprojection only). Build context must contain model/ and tools/.
FROM python:3.13
# ⚠️ Pin zarr's MAJOR version: zarr 2 exposes Group.create_dataset, zarr 3 replaced
# it with create_array, and the pipeline's two tools were written against different
# majors. That split is what tools/zarr_v3_to_v2.py was really working around, and
# it broke the serving build until add_nowcast_channels learned to dispatch.
RUN pip install --no-cache-dir "rasterio==1.4.*" pyproj "zarr==3.*" "numpy==2.*"
COPY model /app/model
COPY tools /app/tools
WORKDIR /app
ENTRYPOINT ["python", "tools/build_seamless_zarr.py"]
