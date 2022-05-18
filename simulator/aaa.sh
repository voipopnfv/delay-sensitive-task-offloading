#!/bin/bash
echo "CONF:"
cat conf.json
echo -e "\nLOG:"
for i in {1..10}
do
   echo 112
    #./record.sh 2>&1 | tee ./log/ControllerComp/$4s/sys/$1_$2_METRIC_$3MB_$4S_NODE3.log &
    #echo $1 $2 $3 $4
   # time go run main.go 2>&1 |tee ./log/ControllerComp/$4s/lat/$1_$2_CONTROLLERCOMP_$3MB_$4S.log;
  #  sleep 0 #20
done
