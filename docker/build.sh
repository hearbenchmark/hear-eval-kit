#!/bin/sh

docker build -t turian/heareval:latest -f docker/Dockerfile-cuda11.2 . && docker push turian/heareval:latest
docker build -t turian/heareval:stable -f docker/Dockerfile-cuda11.2-stable . && docker push turian/heareval:stable
