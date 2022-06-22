#!/usr/bin/env bash

# turn on bash's job control
set -m

python3 Multi-Signbhoard-gpu-grpcserver.py &
sleep 5
python3 init.py
fg %1
