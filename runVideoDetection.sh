#!/bin/bash
killall ffserver
ffserver -d -f ./ffserver.conf &
python ./runVideoDetection.py --input 1 --output 1
