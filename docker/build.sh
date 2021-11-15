#!/bin/sh

docker build -t turian/heareval:latest -f docker/Dockerfile-cuda11.2 . && docker push turian/heareval:latest
docker build -t turian/heareval:latest38 -f docker/Dockerfile-cuda11.2-python3.8 . && docker push turian/heareval:latest38
docker build -t turian/heareval:latestmulti -f docker/Dockerfile-cuda11.2-multipy . && docker push turian/heareval:latestmulti
docker build -t turian/heareval:stable -f docker/Dockerfile-cuda11.2-stable . && docker push turian/heareval:stable
