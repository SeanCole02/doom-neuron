#!/bin/bash
# This will launch the VizDoom environment and RL agent.
# Launch this AFTER the CL1 component.
# (sys-016)     192.168.218.3
# (cl1-2507-15) 192.168.240.84
#
# cl1-host
# Replace cl1-host with ip address of the CL1 device

python training_server.py \
    --mode watch \
    --device cuda \
    --checkpoint checkpoints/l5_2048_rand/final_model.pt \
    --cl1-host 127.0.0.1 \
    --max-episodes 800