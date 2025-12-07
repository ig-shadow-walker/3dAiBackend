# Multi-stage Dockerfile for 3DAIGC-API
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Set CUDA architectures for compilation (needed for building CUDA extensions without GPU present)
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

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

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Setup the proxy of git and pip (MODIFY THIS ACCORDING TO YOUR ENV)
RUN git config --global http.proxy http://127.0.0.1:7890
RUN git config --global https.proxy http://127.0.0.1:7890
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN export http_proxy=http://127.0.0.1:7890
RUN export https_proxy=http://127.0.0.1:7890

# Install TRELLIS dependencies
WORKDIR /app/thirdparty/TRELLIS
RUN bash setup.sh --basic --diffoctreerast --spconv --mipgaussian --nvdiffrast
RUN pip install ../../assets/wheels/utils3d-0.0.2-py3-none-any.whl --no-deps
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# RUN pip install ../../assets/wheels/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
# RUN pip install ../../assets/wheels/kaolin-0.17.0-cp310-cp310-linux_x86_64.whl

# Install PartField dependencies
WORKDIR /app/thirdparty/PartField
RUN pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
RUN pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d psutil
RUN pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_scatter-2.1.2%2Bpt24cu121-cp310-cp310-linux_x86_64.whl
RUN pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_cluster-1.6.3%2Bpt24cu121-cp310-cp310-linux_x86_64.whl
# RUN pip install ../../assets/wheels/torch_scatter-2.1.2+pt24cu121-cp310-cp310-linux_x86_64.whl
# RUN pip install ../../assets/wheels/torch_cluster-1.6.3+pt24cu121-cp310-cp310-linux_x86_64.whl

# Install Hunyuan3D 2.0 dependencies
WORKDIR /app/thirdparty/Hunyuan3D-2
RUN pip install -r requirements.txt
RUN pip install -e .

# RUN export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Build custom rasterizer for Hunyuan3D 2.0
WORKDIR /app/thirdparty/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py install

# Build differentiable renderer for Hunyuan3D 2.0
WORKDIR /app/thirdparty/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py install

# Install Hunyuan3D 2.1 dependencies
WORKDIR /app/thirdparty/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
# RUN pip install -e .
RUN python3 setup.py install

# Build differentiable renderer for Hunyuan3D 2.1
WORKDIR /app/thirdparty/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh

# Install Hunyuan3D 2.1 requirements
WORKDIR /app/thirdparty/Hunyuan3D-2.1
RUN pip install -r requirements-inference.txt

# Install HoloPart dependencies
WORKDIR /app/thirdparty/HoloPart
RUN pip install -r requirements.txt

# Install UniRig dependencies
WORKDIR /app/thirdparty/UniRig
RUN pip install spconv-cu120 pyrender fast-simplification python-box timm

# Install PartPacker dependencies
WORKDIR /app/thirdparty/PartPacker
RUN pip install pybind11==3.0.1
RUN pip install meshiki kiui fpsample pymcubes einops

# Install PartCrafter dependencies (if requirements exist)
WORKDIR /app/thirdparty/PartCrafter
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi

# Install main project dependencies
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -r requirements-test.txt
RUN pip install huggingface_hub

# Create necessary directories
RUN mkdir -p /app/uploads /app/data

# Set environment variables for runtime
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CONDA_DEFAULT_ENV=3daigc-api

# Expose port for FastAPI
EXPOSE 7842
