# nuPlan devkit 1.2.2 + this repo's planner, for linux/arm64 (Jetson Thor). CPU only: nothing here uses the GPU.
# Build from the repo root:  docker build -f docker/Dockerfile -t nuplan-ilqr-mpc .
FROM python:3.9-slim-bookworm

# git: devkit clone. build-essential/cmake: sdists and the P7 C++ core. libgl1/libglib2.0-0: opencv. libgomp1: torch.
# libgdal-dev/gdal-bin: fiona and rasterio have no linux-arm64 wheels for Python 3.9, so they build against Debian's GDAL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake libgl1 libglib2.0-0 libgomp1 libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.12.20

COPY docker/requirements.lock /tmp/requirements.lock
RUN uv pip install --system --no-deps --prerelease=allow -r /tmp/requirements.lock

# Same devkit commit as the Mac (stage1-install-check.md).
ARG DEVKIT_COMMIT=e924167
RUN git clone https://github.com/motional/nuplan-devkit.git /opt/nuplan-devkit \
    && git -C /opt/nuplan-devkit checkout ${DEVKIT_COMMIT} \
    && uv pip install --system --no-deps -e /opt/nuplan-devkit

# tuPlan garage for the PDM-Closed baseline (P4). Only its planner is used: the training-only requirements
# (scikit-learn, positional-encodings) are not needed, so --no-deps leaves the lock unchanged.
ARG GARAGE_COMMIT=b51d5d0
RUN git clone https://github.com/autonomousvision/tuplan_garage.git /opt/tuplan_garage \
    && git -C /opt/tuplan_garage checkout ${GARAGE_COMMIT} \
    && uv pip install --system --no-deps -e /opt/tuplan_garage

# The repo is bind-mounted at /work; data read-only at /data/nuplan; outputs at /data/exp.
ENV NUPLAN_DATA_ROOT=/data/nuplan \
    NUPLAN_MAPS_ROOT=/data/nuplan/maps \
    NUPLAN_EXP_ROOT=/data/exp \
    OPENBLAS_NUM_THREADS=1 \
    PYTHON=python \
    DEVKIT=/opt/nuplan-devkit \
    PYTHONPATH=/work
WORKDIR /work
