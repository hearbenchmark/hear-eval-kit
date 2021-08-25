#docker pull turian/heareval
## Or, build the docker yourself
##docker build -t turian/heareval .
#
#docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -h $HOSTNAME -v $HOME/.Xauthority:/home/renderman/.Xauthority -it turian/heareval bash
#


# deepo: python3 generate.py Dockerfile pytorch tensorflow keras python==3.7
# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.7    (apt)
# pytorch       latest (pip)
# tensorflow    latest (pip)
# keras         latest (pip)
# ==================================================================

FROM ubuntu:18.04

ENV LANG C.UTF-8

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update
    #   apt-get update && \

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        unzip unrar

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && apt-get install -y -q

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install

    #make -j"$(nproc)" install && \

# ==================================================================
# python
# ------------------------------------------------------------------

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.7 \
        python3.7-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.7 ~/get-pip.py && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm

# ==================================================================
# pytorch
# ------------------------------------------------------------------

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        --pre torch torchvision -f \
        https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

#        && \

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL \
        tensorflow

#        && \

# ==================================================================
# keras
# ------------------------------------------------------------------

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL \
        h5py \
        keras

#        && \

# ==================================================================
# heareval
# ------------------------------------------------------------------

## workaround readline fallback
#RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && apt-get install -y -q

RUN apt update
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL software-properties-common
RUN apt update
RUN apt upgrade -y
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL sox

# h5
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL pkg-config libhdf5-100 libhdf5-dev

# LLVM >= 9.0
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL --reinstall python3-apt && \
    $APT_INSTALL gpg-agent
RUN wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
# Might need to change python version: nano /usr/bin/add-apt-repository
#RUN add-apt-repository 'deb http://apt.llvm.org/bionic/   llvm-toolchain-bionic-10  main'
RUN add-apt-repository 'deb http://apt.llvm.org/bionic/   llvm-toolchain-bionic-11  main'
RUN apt update
#RUN $APT_INSTALL llvm-10 lldb-10 llvm-10-dev libllvm10 llvm-10-runtime
#RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-10 10
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL llvm-11 lldb-11 llvm-11-dev libllvm11 llvm-11-runtime
RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-11 11
RUN update-alternatives --config llvm-config


## Need this crap to get python 3.7 (or 3.8)
## Might not need to add deadsnakes, since deepo has it already
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN $APT_INSTALL python3.7 python3.7-dev
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
##RUN $APT_INSTALL python3.8 python3.8-dev
##RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 3
#RUN $APT_INSTALL python3-pip cython
##RUN update-alternatives  --set python3  /usr/bin/python3.7
#
## Is this right?
## https://askubuntu.com/questions/1229095/modulenotfounderror-no-module-named-apt-pkg-after-installing-python-3-7
#RUN update-alternatives --remove-all python3
#RUN ln -sf /usr/bin/python3.6 /usr/bin/python3

# For ffmpeg >= 4.2
# Could also build from source:
# https://github.com/jrottenberg/ffmpeg/blob/master/docker-images/4.3/ubuntu1804/Dockerfile
RUN add-apt-repository ppa:jonathonf/ffmpeg-4
RUN apt-get update
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL ffmpeg

## add to bashrc:
#alias python3=python3.7

##RUN $APT_INSTALL python3-pandas
#RUN python3 -m pip install numpy
##pip3 install cython
##pip3 install -e ".[dev]"


# gsutil
# https://cloud.google.com/storage/docs/gsutil_install#deb
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL apt-transport-https ca-certificates gnupg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL google-cloud-sdk
#gcloud init

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL cython
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL hearbaseline
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $GIT_CLONE https://github.com/neuralaudio/hear-eval-kit.git
RUN cd hear-eval-kit && \
    python -m pip --no-cache-dir install -e ".[dev]"

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL less

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 6006
