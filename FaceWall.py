import cv2
import numpy as np
import time
import dlib
import subprocess as sp

from websocket import create_connection
#Info for FACEWALL
FACEWALL_HEIGHT = 720
FACEWALL_WIDTH = 1280
FACE_DIM = 78
FACE_SPACE = 2
FACE_NROW = 9
FACE_NCOL = 16

#Info for streaming using ffmpeg
FFMPEG_PROC = None;
WIDTH = 1280;
HEIGHT = 720;

FPS = 24;

#Format the video output
ffmpeg = 'ffmpeg'
dimension = '{}x{}'.format(WIDTH, HEIGHT)
f_format = 'bgr24' #OpenCV bgr format, tested rbg24 without success
fps = str(FPS);

output_file = "faceWall.mp4"

class FaceWall():

    def __init__(self, output = output_file):
        FFMPEG_COMMAND =[ffmpeg,
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', dimension,
                '-pix_fmt', f_format,
                '-r', fps,
                '-i', '-',
                '-an',
                '-vcodec', 'mpeg4',
                '-b:v', '5000k',
                output ] 
        FFMPEG_COMMAND = ' '.join(FFMPEG_COMMAND)
        self.FFMPEG_PROC = sp.Popen(FFMPEG_COMMAND, stdin=sp.PIPE, stdout=sp.PIPE, shell=True)
        self.faceCounts = 0
        self.start = time.time()
        self.isRendered = False

        self.faceWall = np.zeros((FACEWALL_HEIGHT, FACEWALL_WIDTH, 3), np.uint8) 
        self.bestFace = np.zeros((FACE_DIM, FACE_DIM, 3), np.uint8)
        self.bestPerson = "None"
        self.bestConfidence = 0.0
        
        self.wsClient = create_connection("ws://127.0.0.1:9000")

    def putNewFaces (self, imgBGR, reps, persons, confidences):
        if (len(confidences)> 0):
            maxConfidence = max(confidences)
            idxBestFace = confidences.index(maxConfidence)
            if (maxConfidence > self.bestConfidence):
                self.bestConfidence = maxConfidence
                bb = reps[idxBestFace][0]
                print(bb.top(), bb.bottom(), bb.left(), bb.right())
                face = imgBGR[bb.top():bb.bottom(), bb.left():bb.right()]
                self.bestFace = cv2.resize(face,(FACE_DIM, FACE_DIM))
                self.bestPerson = persons[idxBestFace]
                print ("Updated new face")
        
    def renderFaces(self):
        elapsedSeconds = int(time.time()-self.start)
        print ("ElapsedSeconds: ", elapsedSeconds)
        if (elapsedSeconds%3 != 2):
            self.isRendered = False
            self.bestConfidence = 0.0
        elif ((elapsedSeconds%3 == 2) and 
                (not self.isRendered)  and 
                (self.bestConfidence > 0)):
            if (self.faceCounts == FACE_NROW * FACE_NCOL):
                self.faceCounts = 0

            faceTop = int(self.faceCounts / FACE_NCOL)*(FACE_DIM+FACE_SPACE)
            faceLeft = (self.faceCounts % int(FACE_NCOL))*(FACE_DIM+FACE_SPACE)
            faceBottom = faceTop + FACE_DIM
            faceRight = faceLeft + FACE_DIM
            print("faceCounts", self.faceCounts)
            print ("Write new face to the wall", (faceTop, faceLeft, faceBottom, faceRight))
            self.faceWall[faceTop:faceBottom, faceLeft:faceRight] = self.bestFace
            self.faceCounts = self.faceCounts + 1

            self.isRendered = True
            self.bestConfidence = 0.0
            if (self.bestPerson!="None"):
                self.wsClient.send(self.bestPerson)
        self.FFMPEG_PROC.stdin.write(self.faceWall.tostring())
        
