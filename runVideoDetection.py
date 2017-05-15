import subprocess as sp
import cv2
import math
import mpipe
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

output_file = "out.mp4"

#Define different worker for mpipe
class FaceDetector(mpipe.OrderedWorker):
    """
       A worker that returns bounding boxes of faces on an img
    """

    def doInit(self):
        self.faceDetection = FaceDetection()

    def doTask(self, params):
        rgbImg = params[0]
        bbs = self.faceDetection.getFaceBBs(rgbImg, multiple=True)
        return (rgbImg, bbs)

class FaceRecognizer(mpipe.OrderedWorker):
    """
        A worker that returns face positions and draw box to images
    """
    
    def doInit(self):
        self.faceDetection = FaceDetection()
    
    def doTask(self, params):
        rgbImg = params[0]
        bbs = params[1]
        reps = self.faceDetection.getReps(rgbImg, bbs, multiple=True)
        if len(reps) > 1:
            print("List of faces in image from left to right")
        persons = []
        confidences = []
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
  
            persons.append(person)
            confidences.append(confidence)
            if self.verbose:
                print("Prediction took {} seconds.".format(time.time() - start))
            if multiple:
                print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'), bbx, confidence))
            else:
                print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
            
        annotatedImg = self.drawBoxes(rgbImg, reps, persons, confidences)
        return (rbgImg, annotatedImg, bbs, persons, confidences)

class PipelineOutput(mpipe.OrderedWorker):
    def doInit(self):
        FFMPEG_COMMAND = ' '.join(FFMPEG_COMMAND)
        self.FFMPEG_PROC = sp.Popen(FFMPEG_COMMAND, stdin=sp.PIPE, 
            stdout=sp.PIPE, shell=True)

    def doTask(self, params):
        rbgImg = params[0]
        annotatedImg = params[1]
        bbs = params[2]
        persons = params[3]
        confidences = params[4]
 
        bgrResult = cv2.cvtColor(annotatedImg, cv2.COLOR_RGB2BGR)
        bgrResult = cv2.resize(bgrResult, (WIDTH, HEIGHT), 
                    interpolation = cv2.INTER_CUBIC )
        
        self.FFMPEG_PROC.stdin.write(bgrResult.toString())

class FaceRecognitionPipeline:
    def __init__(self):
        stage0 = mpipe.Stage(FaceDetector, 7)
        stage1 = mpipe.Stage(FaceRecognizer, 1)
        stage2 = mpipe.Stage(PipelineOutput, 1)
        stage0.link(stage1)
        stage1.link(stage2)
        self.pipe = mpipe.Pipeline(stage0)
    def put(self, rbgImg):
        self.pipe.put(rbgImg)

    def stop(self):
        self.pipe.put(None)

faceRecognitionPipeline = FaceRecognitionPipeline()

cap = cv2.VideoCapture('/root/openface/demos/video/in.mp4')
if(cap.isOpened()):
    for i in range(1,500):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Put to the pipeline for detection
            faceRecognitionPipeline.put(frame_rgb) 
        else:
            break
cap.release()

