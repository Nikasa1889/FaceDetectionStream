import cv2
import sys
import numpy as np

# This FaceDetection class use dlib to detect and align faces,
# use openface to find encoding of the face,
# and use SVM to classify face encodings
#
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontThickness = 1

def drawBox(rgbImg, bb, person, confidence):
    #Draw bounding box first
    cv2.rectangle(rgbImg, 
                (bb.left(), bb.bottom()), (bb.right(), bb.top()),
                (127, 255, 212),
                4)
    #Calculate text length
    textSize = cv2.getTextSize(
                '{}-{:.2f}'.format(person.decode('utf-8'), confidence), 
                    fontFace, fontScale, fontThickness)[0]
    #Draw the text background
    cv2.rectangle(rgbImg, 
                (bb.left(), bb.top()), 
                (bb.left() + textSize[0], bb.top() -textSize[1]), 
                (127, 255, 212), -1);
    #Now put the text on it
    cv2.putText(rgbImg, 
                '{}-{:.2f}'.format(person.decode('utf-8'), confidence), 
                (bb.left(), bb.top()-1),
                fontFace, fontScale,(0, 0, 0), fontThickness
                )
    return rgbImg

def drawBoxes (rgbImg, reps, persons, confidences):
    for r, person, confidence in zip(reps, persons, confidences):
            bb = r[0]
            rgbImg = drawBox(rgbImg, bb, person, confidence)
    return rgbImg
