#!/bin/bash
killall ffserver
ffserver -d -f ./ffserver.conf &
python3 ./runVideoDetection.py --input 0 --output 0
