FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y \
    build-essential \
    wget \
    git \
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
    tk-dev
RUN apt-get install -y --allow-unauthenticated graphviz
RUN apt install -y curl ffmpeg
RUN curl -OL https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tar.xz
RUN tar xJf Python-3.9.10.tar.xz
RUN cd Python-3.9.10 && ./configure && make && make install
RUN pip3 install numpy
RUN pip3 install scikit-video
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install opencv-python
RUN pip3 install nvidia-ml-py3
RUN pip3 install lpips
RUN pip3 install scikit-image
RUN pip3 install tqdm
RUN pip3 install matplotlib
RUN pip3 install imgcat
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
