# syntax=docker/dockerfile:experimental
# https://github.com/gradient-ai/base-container/tree/main/pt211-tf215-cudatk120-py311
# Ubuntu 22.04
# Python 3.10
# PyTorch 2.1.2 (CUDA 11.8)
# CUDA Toolkit 12.0, CUDNN 8.9.7
# ==================================================================
# Initial setup
# ------------------------------------------------------------------

# Ubuntu 22.04 as base image
FROM ubuntu:22.04
# RUN yes| unminimize

# Set ENV variables
ENV LANG C.UTF-8
ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

ENV APT_INSTALL="apt-get install -y --no-install-recommends"

# ==================================================================
# Tools
# ------------------------------------------------------------------

RUN apt-get update && \
    $APT_INSTALL \
    sudo \
    build-essential \
    ca-certificates \
    wget \
    curl \
    git \
    zip \
    unzip \
    nano \
    ffmpeg \
    software-properties-common \
    gnupg \
    python3 \
    python3-pip \
    python3-dev

# Add symlink so python and python3 commands use same python3.9 executable
RUN ln -s /usr/bin/python3 /usr/local/bin/python    

# ==================================================================
# Installing CUDA packages (CUDA Toolkit 12.0 and CUDNN 8.9.7)
# ------------------------------------------------------------------
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    $APT_INSTALL cuda && \  
    rm cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb

# Installing CUDNN
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
    apt-get update && \
    $APT_INSTALL libcudnn8=8.9.7.29-1+cuda12.2  \
    libcudnn8-dev=8.9.7.29-1+cuda12.2


ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN python3 -m pip --no-cache-dir install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY ./requirements.txt /home/app/
WORKDIR /home/app/
RUN python -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm && python -m spacy download ja_core_news_sm
