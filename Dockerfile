# Multi-stage Dockerfile for 3DAIGC-API
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Set CUDA architectures for compilation (needed for building CUDA extensions without GPU present)
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Note: nvidia-smi cannot run during build time, only at runtime
# RUN nvidia-smi

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libegl1 \
    libegl1-mesa \
    libgl1-mesa-dev \
    libjpeg-dev \
    libwebp-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    libxi6 \
    libxkbcommon-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Create conda environment with Python 3.10
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n 3daigc-api python=3.10 -y

# Make conda environment activation persistent
SHELL ["conda", "run", "-n", "3daigc-api", "/bin/bash", "-c"]

# Install PyTorch with CUDA 12.4 support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Setup proxy settings passed via build args
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy
ENV no_proxy=$no_proxy

# Install TRELLIS.2 dependencies
# setup.sh requires nvidia-smi/rocminfo to detect GPU; no GPU available during build, so use a stub
WORKDIR /app/thirdparty/TRELLIS.2
RUN mkdir -p /tmp/build-bin && \
    printf '#!/bin/bash\nexit 0\n' > /tmp/build-bin/nvidia-smi && \
    chmod +x /tmp/build-bin/nvidia-smi && \
    PATH="/tmp/build-bin:$PATH" bash setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
# Ensure flash-attn is installed (setup.sh install may fail during build without GPU; use pre-built wheel)
RUN pip install https://fishwowater.oss-cn-shenzhen.aliyuncs.com/flash_attn-2.7.3-cp310-cp310-linux_x86_64.whl --no-deps
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html

# Install TRELLIS(v1) requirements on top of TRELLIS.2
RUN pip install pymeshfix igraph
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation && \
    rm -rf /tmp/extensions/mip-splatting

# Install PartField dependencies
WORKDIR /app/thirdparty/PartField
RUN pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
RUN pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d psutil
RUN pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html --no-cache-dir

# Install Hunyuan3D 2.1 dependencies
WORKDIR /app/thirdparty/Hunyuan3D-2.1
RUN cd hy3dpaint/custom_rasterizer && pip install -e . --no-build-isolation

# Build differentiable renderer for Hunyuan3D 2.1
WORKDIR /app/thirdparty/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh

# Install Hunyuan3D 2.1 requirements
WORKDIR /app/thirdparty/Hunyuan3D-2.1
RUN pip install -r requirements-inference.txt

# Install UniRig dependencies
WORKDIR /app/thirdparty/UniRig
RUN pip install spconv-cu120 pyrender fast-simplification python-box timm

# Install PartPacker dependencies
WORKDIR /app/thirdparty/PartPacker
RUN pip install pybind11==3.0.1
RUN pip install meshiki kiui fpsample pymcubes einops

# Install PartUV dependencies
RUN pip install seaborn partuv blenderproc

# Install P3-SAM (Hunyuan3D-Part) dependencies
WORKDIR /app/thirdparty/Hunyuan3DPart/P3SAM
RUN pip install numba scikit-learn fpsample

# Install FastMesh dependencies
WORKDIR /app/thirdparty/FastMesh
RUN if [ -f "requirement_extra.txt" ]; then pip install -r requirement_extra.txt; fi

# Install UltraShape dependencies
WORKDIR /app/thirdparty/UltraShape
RUN pip install git+https://github.com/ashawkey/cubvh --no-build-isolation

# Install VoxHammer dependencies
WORKDIR /app/thirdparty/VoxHammer
RUN pip install git+https://github.com/huanngzh/bpy-renderer.git
RUN pip install pysdf sentencepiece

# Install main project dependencies
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -r requirements-test.txt
RUN pip install huggingface_hub

# Create necessary directories
RUN mkdir -p /app/uploads /app/data /app/logs /app/outputs

# Set environment variables for runtime
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CONDA_DEFAULT_ENV=3daigc-api

ENV http_proxy=""
ENV https_proxy=""
ENV no_proxy=""
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV NO_PROXY=""

# Expose port for FastAPI
EXPOSE 7842

# Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#   CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
