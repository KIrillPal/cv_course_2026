# CUDA + dev image: нужен nvcc для сборки MinkowskiEngine.
# PyTorch 2.5 + cu124; ME собирается из исходников (OpenBLAS в образе, без conda MKL).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_BUILD_ISOLATION=1
# Иначе wheel torch (~2GB+) дублируется в кэше pip → часто Errno 28 при сборке образа
ENV PIP_NO_CACHE_DIR=1
# toolkit в devel-образе; нужен для ME + --force_cuda при сборке без GPU
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git wget unzip ca-certificates curl \
    libopenblas-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /workspace

# setuptools<60: numpy.distutils / ME setup.py
RUN python -m pip install --upgrade pip wheel packaging ninja \
    && python -m pip install "setuptools>=59.5,<60" "numpy>=1.21,<2"

RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Архитектуры GPU (можно переопределить при build: --build-arg TORCH_CUDA_ARCH_LIST="8.0")
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=8

# pip install ME без GPU → CPU_ONLY. Собираем через setup.py --force_cuda (см. ME setup.py).
RUN git clone --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git /opt/MinkowskiEngine \
    && cd /opt/MinkowskiEngine \
    && python setup.py install --blas=openblas --force_cuda \
    && rm -rf /opt/MinkowskiEngine

RUN python -m pip install h5py scikit-learn matplotlib \
    && python -c "import torch, MinkowskiEngine as ME; print('torch', torch.__version__, 'ME', ME.__version__, 'ME.is_cuda_available()=', ME.is_cuda_available())"

COPY train_modelnet40.py /workspace/train_modelnet40.py

ENV LOG_DIR=/logs
ENV WORKDIR=/workspace
ENV MAX_STEPS=5000
ENV VAL_FREQ=500
ENV STAT_FREQ=50
ENV BATCH_SIZE=32

CMD ["python", "-u", "/workspace/train_modelnet40.py"]
