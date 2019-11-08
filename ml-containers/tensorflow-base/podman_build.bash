#!/bin/bash
DOCKER_REGISTRY=docker.io
DOCKER_PROJECT_ID=docker.io/djkormo
SERVICE_NAME=jetson-tensorflow-base:0.1.0
DOCKER_IMAGE_NAME=$DOCKER_PROJECT_ID/$SERVICE_NAME
DOCKER_IMAGE_REPO_NAME=$DOCKER_PROJECT_ID/$SERVICE_NAME

echo "DOCKER_REGISTRY: $DOCKER_REGISTRY"
echo "DOCKER_PROJECT_ID: $DOCKER_PROJECT_ID"
echo "SERVICE_NAME: $SERVICE_NAME"
echo "DOCKER_IMAGE_NAME: $DOCKER_IMAGE_NAME"
echo "DOCKER_IMAGE_REPO_NAME: $DOCKER_IMAGE_REPO_NAME"

#  build

podman build -t $DOCKER_PROJECT_ID/$SERVICE_NAME . -f Dockerfile

# tag

#podman tag $SERVICE_NAME $DOCKER_IMAGE_NAME

#push

podman push  $DOCKER_PROJECT_ID/$SERVICE_NAME

























 1Help              2Save              3Mark               4Replac             5Copy               6Move              7Search             8Delete             9PullDn            10Quit
