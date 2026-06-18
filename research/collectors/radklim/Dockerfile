# RADKLIM collector — wradlib reads RADOLAN binary; wradlib needs GDAL/HDF5 which
# don't pip-install cleanly, so use conda-forge via micromamba. Build context =
# research/collectors.
FROM mambaorg/micromamba:1.5.8
RUN micromamba install -y -n base -c conda-forge \
      python=3.12 wradlib scipy rasterio numpy && micromamba clean -a -y
ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER fetch_radklim.py /app/fetch_radklim.py
WORKDIR /app
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "fetch_radklim.py"]
