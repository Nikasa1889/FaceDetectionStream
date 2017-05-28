# FaceDetectionStream
Realtime face detection with ffserver streaming output to local network

This code runs out of the box on this [preconfigured virtual machine](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b)

### VM Configuration:
- You should maximize the number of cores to be used by the image, as long as installing VMware tools on it. That can help increase the speed very much.
- Install ffmpeg3 by running this:
  ```bash
  sudo add-apt-repository ppa:jonathonf/ffmpeg-3
  sudo apt update && sudo apt install ffmpeg libav-tools x264 x265
  ```
- Install twisted, autobahn, and websocket-client to host a simple echo server
  ```bash
  sudo pip install twisted
  sudo pip install autobahn
  sudo pip install websocket-client
  ```
### How to run:
  ```bash
  cd $FaceDetectionStream
  ./runVideoDetection.sh
  ```
