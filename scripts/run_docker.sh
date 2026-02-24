#!/bin/bash

# Get current user ID and group ID
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Mounts the ports required for UDP streaming, see USAGE.md
sudo docker run -it \
  -v .:/root \
  -p 12345:12345 \
  -p 12346:12346 \
  -p 12347:12347 \
  -p 12348:12348 \
  -p 12349:12349 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  --user ${USER_ID}:${GROUP_ID} \
  rocm/pytorch/seandoom:latest