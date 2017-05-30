import cv2
import sys
import numpy as np

# This FaceDetection class use dlib to detect and align faces,
# use openface to find encoding of the face,
# and use SVM to classify face encodings
#
fontFace = cv2.FONT_HERSHEY_DUPLEX
fontThickness = 1
def drawText(rgbImg, text, x, y, fontScale = 0.6):
    #Calculate text length
    textSize = cv2.getTextSize(
                text, 
                fontFace, fontScale, fontThickness)[0]
    #Draw the text background
    cv2.rectangle(rgbImg, 
                (x, y), 
                (x + textSize[0], y -textSize[1]), 
                (0, 0, 255), cv2.FILLED);
    #Now put the text on it
    cv2.putText(rgbImg, 
                text, 
                (x, y-2),
                fontFace, fontScale,(255, 255, 255), fontThickness
                )
    return rgbImg 

def drawBox(rgbImg, bb, person, confidence):
    #Draw bounding box first
    cv2.rectangle(rgbImg, 
                (bb.left(), bb.bottom()), (bb.right(), bb.top()),
                (0, 0, 255),
                4)
    text = '{}-{:.2f}'.format(person, confidence)
    rgbImg = drawText(rgbImg, text, bb.left(), bb.top())
    return rgbImg

def drawBoxes (rgbImg, reps, persons, confidences):
    for r, person, confidence in zip(reps, persons, confidences):
            bb = r[0]
            rgbImg = drawBox(rgbImg, bb, person, confidence)
    return rgbImg
