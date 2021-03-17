FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    llvm-10 \
    llvm-10-dev \
    git \
    libsndfile1
ENV LLVM_CONFIG /usr/bin/llvm-config-10
RUN pip3 install setuptools wheel
RUN pip3 install \
    numpy==1.20.1 \
    scipy==1.6.1 \
    matplotlib==3.3.4 \
    numba==0.53.0 \
    librosa==0.8.0 \
    ipython==7.21.0 \
    torch==1.4.0 \
    tensorboard==2.4.1

RUN git clone https://github.com/zlin7/CGNet
RUN pip3 --no-cache-dir install CGNet/CGNet/cudaCG/cuda
WORKDIR /workspace
ENV NUMBA_CACHE_DIR /tmp/
