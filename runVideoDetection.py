import subprocess as sp
import cv2
import math
#from FaceDetectionOpenFace import FaceDetection
from FaceDetectionDlib import FaceDetection
#from FaceDetectionOpenFace import FaceDetection as Util
from Utils import drawBoxes
from FaceWall import FaceWall
import dlib
import sys
import argparse

#Info for face detection
SKIP_FRAMES = 3
DOWNSAMPLE_RATIO = 1.0
CROP_X = 370 
CROP_Y = 20
CROP_WIDTH = 700
CROP_HEIGHT = 700
#Info for streaming using ffmpeg
FFMPEG_PROC = None;
WIDTH = 1280;
HEIGHT = 720;

FPS = 24;

#Format the video output
ffmpeg = 'ffmpeg'
dimension = '{}x{}'.format(WIDTH, HEIGHT)
f_format = 'bgr24' #OpenCV bgr format, tested rbg24 without success
#f_format = 'rgb24'
fps = str(FPS);


if __name__ == '__main__':
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--input',
            type=int,
            default=0,
            help="Webcam ID, if = -1, it will use in.mov video as input"
            )
    parser.add_argument(
            '--output', 
            type=int, 
            default=0,
            help="0 to output to video, and 1 to live stream video"
            )

    args = parser.parse_args()
    if (args.input == -1):
        cap = cv2.VideoCapture('./in.mov')
        cap.set(1, 1000) #begin at frame 200
    else:
        cap = cv2.VideoCapture(args.input)
	#ret = cap.set(cv2.cv.CV_CAP_PROP_FPS, 12)
        print ("Camera Width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print ("Camera Height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        #if not ret:
        #    print("===========================================================")
        #    print("Warning: Can't change the resolution of the webcam to 720p")
        #    print("===========================================================")
        WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (WIDTH == 640) and (HEIGHT == 480):
            DOWNSAMPLE_RATIO = 1.0;
            CROP_X = 0; CROP_Y = 0; 
            CROP_WIDTH = WIDTH-1; CROP_HEIGHT = HEIGHT-1;

    if (args.output == 0):
        faceDetectionOutput = "out.mp4"
        faceWallOutput = "faceWall.mp4"
    else:
        if (WIDTH == 640) and (HEIGHT == 480):
            faceDetectionOutput = "http://localhost:8000/feed1_480p.ffm"
        else:
            faceDetectionOutput = "http://localhost:8000/feed1_720p.ffm"
        faceWallOutput = "http://localhost:8000/feed2_720p.ffm"
    
    faceDetection = FaceDetection()
    faceWall = FaceWall(output = faceWallOutput)
    dimension = '{}x{}'.format(WIDTH, HEIGHT)
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
                faceDetectionOutput ] 

    FFMPEG_COMMAND = ' '.join(FFMPEG_COMMAND)

    FFMPEG_PROC = sp.Popen(FFMPEG_COMMAND, stdin=sp.PIPE, 
        stdout=sp.PIPE, shell=True)
       
    count = 0
    if(cap.isOpened()):
        reps = []
        persons = []
        confidences = []
        #for i in range(2000):
        while True:
            ret, frame = cap.read()
            if ret:
                #if (frame.shape[0]!=1280) or (frame.shape[1]!=720):
                #    frame = cv2.resize(frame, (1280, 720))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #Scale to smaller image
                if (DOWNSAMPLE_RATIO != 1.0):
                    frame_small = cv2.resize(frame_rgb, (0, 0), fx=1/DOWNSAMPLE_RATIO, fy=1/DOWNSAMPLE_RATIO)
                else:
                    frame_small = frame_rgb
                frame_small = frame_small[CROP_Y:(CROP_Y+CROP_HEIGHT), CROP_X:(CROP_X+CROP_WIDTH)].copy() 
                #Detection here
                if (count % SKIP_FRAMES == 0):
                    (reps, persons, confidences) = faceDetection.infer(frame_small)
                adjustedReps = []
                for rep in reps:
                    bb = rep[0]
                    rep = rep[1]
                    bb = dlib.rectangle(
                        left=int((bb.left()+CROP_X)*DOWNSAMPLE_RATIO),
                        top=int((bb.top()+CROP_Y)*DOWNSAMPLE_RATIO),
                        right=int((bb.right()+CROP_X)*DOWNSAMPLE_RATIO),
                        bottom=int((bb.bottom()+CROP_Y)*DOWNSAMPLE_RATIO))
                    adjustedReps.append((bb, rep))

                faceWall.putNewFaces(frame, adjustedReps, persons, confidences)
                frame_rgb = drawBoxes(frame_rgb, adjustedReps, persons, confidences)      
            
                #Output
                result = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                #result = cv2.resize(result, (WIDTH, HEIGHT), 
                #        interpolation = cv2.INTER_CUBIC )
                FFMPEG_PROC.stdin.write(result.tostring())
                faceWall.renderFaces()
                count = count+1
                if (count > 20000):
                    count = 0
            else:
                break
    cap.release()


