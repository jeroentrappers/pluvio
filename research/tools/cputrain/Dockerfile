# CPU training image for hetz1 (no GPU). torch CPU wheel + the dataset deps
# (zarr 3 reads the v3 store directly — no v2 conversion; pyproj because the
# dataset imports model.geo). Model/tools code is mounted at run time. Thread
# caps + `nice` + docker --cpus at run time keep it off the live backend's back.
FROM python:3.13-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy zarr pyproj
WORKDIR /app
