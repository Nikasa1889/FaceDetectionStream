#!/bin/bash
killall ffserver
killall python3
ffserver -d -f ./ffserver.conf &
python3 ./echoServer.py &
python3 ./runVideoDetection.py --input -1 --output 1
