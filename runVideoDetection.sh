#!/bin/bash
killall ffserver
killall python3
python3 UpdateJsMessages.py --welcomemesfile welcomeMessages.txt --jsfilein app_template.js --jsfileout app.js
ffserver -d -f ./ffserver.conf &
python3 ./echoServer.py &
python3 ./runVideoDetection.py --input 0 --output 1
