#!/bin/bash
# This will launch the VizDoom environment and RL agent.
# Launch this AFTER the CL1 component.
# (sys-016)     192.168.218.3
# (cl1-2507-15) 192.168.240.84
#
# cl1-host 
# Replace cl1-host with ip address of the CL1 device

python training_server.py \
    --mode train \
    --device cuda \
    --cl1-host 192.168.240.84 \
    --max-episodes 400