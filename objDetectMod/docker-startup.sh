#!/usr/bin/env bash

# turn on bash's job control
set -m

python3 object_detection_app-grpcserver.py &
sleep 5
python3 init.py
fg %1
