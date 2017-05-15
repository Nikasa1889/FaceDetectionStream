import subprocess as sp
import sys
import os
import pickle
import cv2
import math
import mpipe
import time
import numpy as np
import pandas as pd
from FaceDetection import FaceDetection
'Info for streaming using ffmpeg'
FFMPEG_PROC = None;
WIDTH = 640;
HEIGHT = 360;

FPS = 30;

#Format the video output
ffmpeg = 'ffmpeg'
dimension = '{}x{}'.format(WIDTH, HEIGHT)
f_format = 'bgr24' #OpenCV bgr format, tested rgb24 without success
#f_format = 'rgb24'
fps = str(FPS);
output_file = "out.mp4"
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
                output_file ] 

#Define different worker for mpipe
class FaceDetector(mpipe.OrderedWorker):
    """
       A worker that returns bounding boxes of faces on an img
    """

    def doInit(self):
        self.faceDetection = FaceDetection(cuda=True)

    def doTask(self, rgbImg):
        bbs = self.faceDetection.getFaceBBs(rgbImg, multiple=True)
        print("Stage 0: detecting face")
        return (rgbImg, bbs)

class FaceRecognizer(mpipe.OrderedWorker):
    """
        A worker that returns face positions and draw box to images
    """
    
    def doInit(self):
        self.faceDetection = FaceDetection(cuda=True)
        with open(self.faceDetection.classifierModel, 'rb') as f:
            if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
            else:
                (le, clf) = pickle.load(f, encoding='latin1')
        self.clf = clf
        self.le = le
    def doTask(self, params):
        print("Stage 1: recognizing faces")
        rgbImg = params[0]
        bbs = params[1]
        reps = self.faceDetection.getReps(rgbImg, bbs, multiple=True)
        persons = []
        confidences = []
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = self.clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = self.le.inverse_transform(maxI)
            confidence = predictions[maxI]
  
            persons.append(person)
            confidences.append(confidence)
            print("Prediction took {} seconds.".format(time.time() - start))
        annotatedImg = self.faceDetection.drawBoxes(
                rgbImg, reps, persons, confidences)
        return (rgbImg, annotatedImg, bbs, persons, confidences)

class PipelineOutput(mpipe.OrderedWorker):
    def doInit(self):
        global FFMPEG_COMMAND
        self.FFMPEG_COMMAND = ' '.join(FFMPEG_COMMAND)
        self.FFMPEG_PROC = sp.Popen(self.FFMPEG_COMMAND, stdin=sp.PIPE, 
            stdout=sp.PIPE, shell=True)

    def doTask(self, params):
        print("Stage 2: outputing stream")
        global WIDTH, HEIGHT
        rgbImg = params[0]
        annotatedImg = params[1]
        bbs = params[2]
        persons = params[3]
        confidences = params[4]
        
        bgrResult = cv2.cvtColor(annotatedImg, cv2.COLOR_RGB2BGR)
        bgrResult = cv2.resize(bgrResult, (WIDTH, HEIGHT), 
                    interpolation = cv2.INTER_CUBIC )
        
        self.FFMPEG_PROC.stdin.write(bgrResult.tostring())
        return None

class FaceRecognitionPipeline:
    def __init__(self):
        stage0 = mpipe.Stage(FaceDetector, 2)
        stage1 = mpipe.Stage(FaceRecognizer, 1)
        stage2 = mpipe.Stage(PipelineOutput, 1)
        stage0.link(stage1)
        stage1.link(stage2)
        self.pipe = mpipe.Pipeline(stage0)

    def put(self, rgbImg):
        print("-------New Image------")
        self.pipe.put(rgbImg)

    #def stop(self):
    #    self.pipe.put(None)

faceRecognitionPipeline = FaceRecognitionPipeline()
#Test straight pipeline
#CON_FFMPEG_COMMAND = ' '.join(FFMPEG_COMMAND)
#FFMPEG_PROC = sp.Popen(CON_FFMPEG_COMMAND, stdin=sp.PIPE, 
#            stdout=sp.PIPE, shell=True)


cap = cv2.VideoCapture('/root/openface/demos/stream/in.mp4')
if(cap.isOpened()):
    for i in range(1,500):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Test straight pipeline
            #bgrResult = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            #bgrResult = cv2.resize(bgrResult, (WIDTH, HEIGHT), 
            #        interpolation = cv2.INTER_CUBIC )
        
            #FFMPEG_PROC.stdin.write(bgrResult.tostring())
 
            #Put to the pipeline for detection
            faceRecognitionPipeline.put(np.asarray(frame_rgb))
        else:
            break
cap.release()

