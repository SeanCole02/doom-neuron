#!/bin/bash
# This will launch the CL1 component of Doom
#
# training-host
# Replace training-host with ip address of the Training Server
# 192.168.1.238 is "geodude"
#
# tick-frequency
# For now, we don't want to overstimulate the neurons, so tick-frequency of 10 Hz
# The game environment runs at a multiple of the tick-frequency

python cl1_neural_interface.py \
    --training-host 192.168.1.238 \
    --recording-path /data/recordings/seandoom/ \
    --tick-frequency 10