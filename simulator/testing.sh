#!/bin/bash
echo 123
#wondershaper clear enp3s0
#wondershaper enp3s0 1024 1024
#./run.sh OBJECT CLOUD 1 4

#sudo wondershaper clear enp3s0
#sudo wondershaper enp3s0 5120 5120
#./run.sh OBJECT CLOUD 5 4

#wondershaper clear enp3s0
#wondershaper enp3s0 10240 10240
#./run.sh OBJECT CLOUD 10 4

#wondershaper clear cnp3s0
#wondershaper enp3s0 1024 1024
#./run.sh VISUAL CONTROLLER 1 4

#wondershaper clear enp3s0
#wondershaper enp3s0 5120 5120
#./run.sh VISUAL CONTROLLER 5 4

wondershaper clear enp3s0
wondershaper enp3s0 10240 10240
sh ./run.sh VISUAL CLOUD 10 1

