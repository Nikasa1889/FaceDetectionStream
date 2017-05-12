import subprocess as sp
import cv2
import math
from FaceDetection import FaceDetection
'Info for streaming using ffmpeg'
FFMPEG_PROC = None;
WIDTH = 640;
HEIGHT = 360;

FPS = 30;

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
cap = cv2.VideoCapture('/root/openface/demos/video/in.mp4')
if(cap.isOpened()):
    frames =  []
    for i in range(1,500):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Detection here
            frame_rgb = faceDetection.infer(frame_rgb)
            
            #Output
            result = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            result = cv2.resize(result, (WIDTH, HEIGHT), 
                    interpolation = cv2.INTER_CUBIC )
            FFMPEG_PROC.stdin.write(result.tostring())
        else:
            break
cap.release()

