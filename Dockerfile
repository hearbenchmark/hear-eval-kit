#docker pull turian/heareval
## Or, build the docker yourself
##docker build -t turian/heareval .
#
#docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -h $HOSTNAME -v $HOME/.Xauthority:/home/renderman/.Xauthority -it turian/heareval bash

FROM ufoym/deepo

## python:3.7.7-stretch
#ENV PATH /usr/local/bin:$PATH

ENV LANG C.UTF-8
# workaround readline fallback
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && apt-get install -y -q

RUN apt-get update

RUN apt update
RUN apt install -y software-properties-common
RUN apt update
RUN apt upgrade -y
RUN apt install -y sox

# h5
RUN apt-get install -y pkg-config libhdf5-100 libhdf5-dev

# LLVM >= 9.0
RUN apt-get install -y --reinstall python3-apt
RUN wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
# Might need to change python version: nano /usr/bin/add-apt-repository
#RUN add-apt-repository 'deb http://apt.llvm.org/bionic/   llvm-toolchain-bionic-10  main'
RUN add-apt-repository 'deb http://apt.llvm.org/bionic/   llvm-toolchain-bionic-11  main'
RUN apt update
#RUN apt install -y llvm-10 lldb-10 llvm-10-dev libllvm10 llvm-10-runtime
#RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-10 10
RUN apt install -y llvm-11 lldb-11 llvm-11-dev libllvm11 llvm-11-runtime
RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-11 11
RUN update-alternatives --config llvm-config


# Need this crap to get python 3.7 (or 3.8)
# Might not need to add deadsnakes, since deepo has it already
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.7 python3.7-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
#RUN apt install python3.8 python3.8-dev
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 3
RUN apt install -y python3-pip cython
#RUN update-alternatives  --set python3  /usr/bin/python3.7

# Is this right?
# https://askubuntu.com/questions/1229095/modulenotfounderror-no-module-named-apt-pkg-after-installing-python-3-7
RUN update-alternatives --remove-all python3
RUN ln -sf /usr/bin/python3.6 /usr/bin/python3

# For ffmpeg >= 4.2
# Could also build from source:
# https://github.com/jrottenberg/ffmpeg/blob/master/docker-images/4.3/ubuntu1804/Dockerfile
RUN add-apt-repository ppa:jonathonf/ffmpeg-4
RUN apt-get update
RUN apt-get install -y ffmpeg

## add to bashrc:
#alias python3=python3.7

#RUN apt install -y python3-pandas
RUN python3 -m pip install numpy
#pip3 install cython
#pip3 install -e ".[dev]"


# gsutil
# https://cloud.google.com/storage/docs/gsutil_install#deb
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get install -y apt-transport-https ca-certificates gnupg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update
RUN apt-get install -y google-cloud-sdk
#gcloud init

RUN python3.7 -m pip install cython
RUN python3.7 -m pip install hearbaseline
RUN git clone https://github.com/neuralaudio/hear-eval-kit.git
RUN cd hear-eval-kit && python3.7 -m pip install -e ".[dev]"
