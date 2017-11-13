import cv2
import numpy as np
import time
import dlib
import subprocess as sp

from websocket import create_connection
from Utils import drawText
#Info for FACEWALL
FACEWALL_HEIGHT = 720
FACEWALL_WIDTH = 1280
FACE_DIM = 156
FACE_SPACE = 4
FACE_NROW = 4
FACE_NCOL = 8

#Info for google-tts command
TTS_COMMAND = "python3 ./simple-google-tts/pyglet_gtts.py"
MESSAGE_FILE = "./welcomeMessages.txt"
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
        self.hasReset = False

        self.faceWall = np.zeros((FACEWALL_HEIGHT, FACEWALL_WIDTH, 3), np.uint8) 
        self.welcomeMessages = []
        with open(MESSAGE_FILE, "r") as f:
            self.welcomeMessages = eval(f.read().encode("utf-8"))
        for idx, person in enumerate(self.welcomeMessages):
            print("{0}   : {1}".format(person, idx))
        self.realtimeFaces = dict.fromkeys(self.welcomeMessages.keys(),None)
        self.countFaces =  dict.fromkeys(self.welcomeMessages.keys(),0)
        self.wsClient = create_connection("ws://127.0.0.1:9000")
        self.announcer = None
        self.waitingMessages = []

    def putNewFaces (self, imgBGR, reps, persons, confidences):
        #Increase the number of occurrence of a person face in the last 3s
        #If within 3s there are more than 10 occurence of that person, then
        #It's his face, pick the best face of him to show first
        for idx, person in enumerate(persons):
            if (person not in self.countFaces.keys()):
                if (person != "Unknown"):
                    print("Detected: {0} , who is not in the list".format(person))
                continue
            #Do not count for detected person
            print("Detected: {0} ".format(person))
            if (self.realtimeFaces[person] is None):
                self.countFaces[person] = self.countFaces[person] + 1
              
            if self.realtimeFaces[person] is not None:
                bb = reps[idx][0]
                face = imgBGR[bb.top():bb.bottom(), bb.left():bb.right()]
                self.realtimeFaces[person] = cv2.resize(face,(FACE_DIM, FACE_DIM), interpolation = cv2.INTER_NEAREST) #Nearest is fastest
        self.renderFaces()
        
    def renderFaces(self):
        print('Entering render faces ...')
        THRESHOLD_FACE_COUNTS = 3
        elapsedSeconds = int(time.time()-self.start)
        if (elapsedSeconds % 3 != 2):
            self.hasReset = False
        elif ((elapsedSeconds % 3 == 2) and 
                (not self.hasReset)):
            print('Reset in renderFaces')
            self.hasReset = True
            person = max(self.countFaces, key = self.countFaces.get)
            if ((self.countFaces[person] > THRESHOLD_FACE_COUNTS) and 
                (self.realtimeFaces[person] is None)):
                #First time detected a new person
                self.realtimeFaces[person] = True #A placeholder for real image
                print('Sending person to ws 9000 ...')
                self.wsClient.send(person)
                print('Sent')
                self.waitingMessages.append(self.welcomeMessages[person])
           #Reset Counting
            for person in self.countFaces.keys():
                self.countFaces[person] = 0
        for idx, (person, face) in enumerate(self.realtimeFaces.items()):
            if (face is not None):
                faceTop = int(idx / FACE_NCOL)*(FACE_DIM+FACE_SPACE)
                faceLeft = (idx % int(FACE_NCOL))*(FACE_DIM+FACE_SPACE)
                faceBottom = faceTop + FACE_DIM
                faceRight = faceLeft + FACE_DIM
                self.faceWall[faceTop:faceBottom, faceLeft:faceRight] = face
                faceWall = drawText(self.faceWall, person, faceLeft, faceBottom+FACE_SPACE, fontScale = 0.5)
        #Announce if there are waiting messages
        try:
            if ((len(self.waitingMessages)>0) and 
                ((self.announcer is None) or 
                (self.announcer.poll() is not None))):
                message = self.waitingMessages.pop(0)
                self.announcer = sp.Popen(['python3', './simple-google-tts/pyglet_gtts.py',"no", message], stdin=sp.PIPE, stdout=sp.PIPE)
        except Exception as e:
            print("Error while running TTS. {0}:{1} ".format(e.errno, e.strerror))
        #Render Facewall
        self.FFMPEG_PROC.stdin.write(self.faceWall.tostring())

