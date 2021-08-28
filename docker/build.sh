#!/bin/sh

docker build -t turian/heareval:cuda10.2 -f Dockerfile-cuda10.2 . && docker push turian/heareval:cuda10.2
docker build -t turian/heareval:cuda11.2 -t turian/heareval:latest -f Dockerfile-cuda11.2 . && docker push turian/heareval:cuda11.2 && docker push turian/heareval:latest
