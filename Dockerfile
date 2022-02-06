FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
USER root
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libbz2-dev \
    libdb-dev \
    libreadline-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libssl-dev \
    zlib1g-dev \
    uuid-dev \
    tk-dev &&\
    apt-get update &&\
    apt-get install -y git &&\
    apt-get install -y --allow-unauthenticated graphviz &&\
    apt install -y curl
RUN curl -OL https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tar.xz
RUN tar xJf Python-3.9.10.tar.xz
RUN cd Python-3.9.10 && ./configure && make && make install
RUN pip3 install -U pip
RUN pip3 install numpy
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install opencv-python
RUN pip3 install nvidia-ml-py3
RUN pip3 install lpips
RUN pip3 install scikit-image
RUN pip3 install tqdm
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
