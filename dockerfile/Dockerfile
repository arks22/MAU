FROM nvidia/cuda:11.6.1-devel-ubuntu18.04
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
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
    tk-dev &&\
    apt-get update &&\
    apt-get install -y --allow-unauthenticated graphviz &&\
    apt install -y curl
RUN curl -OL https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tar.xz
RUN tar xJf Python-3.9.10.tar.xz
RUN cd Python-3.9.10 && ./configure && make && make install
WORKDIR /tmp
RUN wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
RUN tar Jxvf ./ffmpeg-release-amd64-static.tar.xz
RUN cp ./ffmpeg*amd64-static/ffmpeg /usr/local/bin/
RUN pip3 install -U pip
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
WORKDIR /home
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
