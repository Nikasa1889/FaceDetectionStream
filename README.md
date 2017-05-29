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
### How to connect to the output streams:
  The realtime face detection stream and the faceWall can be viewed at:
  ```
  localhost:8000/stat.html
  ```
  The echo server that keeps announcing newly detected faces is hosted at:
  ```
  localhost:8080
  ```
  To let other machine to access to the services hosted inside VMware, you must set the network of the virtual machine to use VMNet0 interface. That is a bridged interface. Then go to Edit -> Virtual Network Editor -> Change Settings (bottom right) -> Select VMNet0 -> in the Bridge to box, select the network apdater that you want to bridge to (the one that the host machine is using to connect to the local network). 
