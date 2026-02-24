#!/bin/bash
# This will build a docker image, run this at project root

IMAGE_NAME="rocm/pytorch/seandoom"
IMAGE_TAG="latest"
DOCKER_FILE="./docker/Dockerfile.rocm"

# Build the docker image from /docker/Dockerfile.rocm
printf "Image name (currently: $IMAGE_NAME): "
read -r IMAGE_NAME_NEW
if [[ -n "$IMAGE_NAME_NEW" ]]; then
    IMAGE_NAME="$IMAGE_NAME_NEW"
fi

printf "Image tag (currently: $IMAGE_TAG): "
read -r IMAGE_TAG_NEW
if [[ -n "$IMAGE_TAG_NEW" ]]; then
    IMAGE_TAG="$IMAGE_TAG_NEW"
fi

printf "Docker file (currently: $DOCKER_FILE): "
read -r DOCKER_FILE_NEW
if [[ -n "$DOCKER_FILE_NEW" ]]; then
    DOCKER_FILE="$DOCKER_FILE_NEW"
fi

printf "\033[33mThis will build the docker image, (y) to continue: \033[0m"
read -r CONTINUE
if [[ "$CONTINUE" =~ ^[Yy]$ ]]; then
    sudo docker build -t $IMAGE_NAME:$IMAGE_TAG -f $DOCKER_FILE .
fi
