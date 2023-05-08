# docker build -t hydra-pl-rscd:v1 .
ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"
ARG MMCV="1.4.6"
# 1.9.0-cuda10.2-cudnn7-runtime decel devel

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

#  pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel   cuda10.2只存在 torch1.9官方版本
# 但是为了使用 新的torchvision功能，必须1.10版本以上的torch


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# # 见加国内镜像 ubuntu20.04
# ADD sources.list /etc/apt 

#  RUN: 在Build构建时执行命令 ENTRYPOINT ：容器启动时执行命令 CMD： 容器启动后执行默认的命令或参数
# libgl1-mesa-glx 解决 libGL.so.1: cannot open shared object file: No such file or directory
# libglib2.0-dev  解决 `libgthread-2.0.so.0: cannot open shared object file: 需要时区信息，待解决
# RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all


ADD oh-my-tuna.py /workspace
ADD requirements.txt /workspace

RUN cd /workspace

# Install PYTORCH_LIghtning + hydra
RUN python oh-my-tuna.py

# RUN ["/bin/bash", "-c", "rm -rf ~/.cache/pip/*"]
# RUN ["/bin/bash", "-c", "python -m pip install --upgrade pip"]
# RUN pip install -r requirements.txt

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install openmim"]
RUN ["/bin/bash", "-c", "mim install mmengine"]
RUN ["/bin/bash", "-c", "mim install mmcv==${MMCV}"]

# Install MMSegmentation
RUN ["/bin/bash", "-c", "mim install mmsegmentation==v0.29.1"]
 