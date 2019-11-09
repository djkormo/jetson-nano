#!/bin/bash
CONTAINER_ID=$(docker ps -qf "ancestor=djkormo/jetson-pytorch-base:0.1.0")
docker commit $CONTAINER_ID djkormo/jetson-pytorch-base:0.2.0