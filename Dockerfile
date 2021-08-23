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

RUN apt-get updat
