#!/bin/bash
while true
do
    python3 record.py   
    a=$?
    if [ "$a" -eq 43 ];
    then
	    echo DONE
	    break
    fi
    sleep 1
done
