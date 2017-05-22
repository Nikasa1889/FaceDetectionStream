import subprocess as sp
import cv2
import math
from FaceDetection import FaceDetection
#Info for face detection
SKIP_FRAMES = 3
DOWNSAMPLE_RATIO = 2.0
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

FFMPEG_COMMAND = ' '.join(FFMPEG_COMMAND)

FFMPEG_PROC = sp.Popen(FFMPEG_COMMAND, stdin=sp.PIPE, 
        stdout=sp.PIPE, shell=True)

faceDetection = FaceDetection()
cap = cv2.VideoCapture('/root/openface/demos/stream/in.mov')
count = 0
if(cap.isOpened()):
    reps = []
    persons = []
    confidences = []
    for i in range(1,10000):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Scale to smaller image
            frame_small = cv2.resize(frame_rgb, (0, 0), fx=1/DOWNSAMPLE_RATIO, fy=1/DOWNSAMPLE_RATIO)
            
            #Detection here
            if (count % SKIP_FRAMES == 0):
               (reps, persons, confidences) = faceDetection.infer(frame_small)
            frame_rgb = faceDetection.drawBoxes(frame_rgb, reps, persons, confidences, DOWNSAMPLE_RATIO)      
            
            #Output
            result = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            #result = cv2.resize(result, (WIDTH, HEIGHT), 
            #        interpolation = cv2.INTER_CUBIC )
            FFMPEG_PROC.stdin.write(result.tostring())
            count = count+1
            if (count > 20000):
                count = 0
        else:
            break
cap.release()


