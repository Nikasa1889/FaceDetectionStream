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
                (x + textSize[0]+2, y - textSize[1]-6), 
                (0, 0, 255), cv2.FILLED);
    #Now put the text on it
    cv2.putText(rgbImg, 
                text, 
                (x+2, y-6),
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


def drawFaceLine(rgbImg, reps, persons, confidences, face_landmarks):
    for r, person, confidence, face_landmark in zip(reps, persons, confidences, face_landmarks):
        bb = r[0]
        #  draw face lines
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]
        for facial_feature in facial_features:
            points = face_landmark[facial_feature]
            for idx, point_1 in enumerate(points):
                if idx < len(points) - 1:
                    point_2 = points[idx + 1]
                    cv2.line(rgbImg, point_1, point_2, color=(224, 224, 224))
        #  draw text
        text = '{}-{:.2f}'.format(person, confidence)
        rgbImg = drawText(rgbImg, text, bb.left(), bb.top())
    return rgbImg


def drawContinuousLines(rgbImg, points, color, thickness=1):
    for idx, point_1 in enumerate(points):
        if idx < len(points) - 1:
            point_2 = points[idx + 1]
            cv2.line(rgbImg, point_1, point_2, color=color, thickness=thickness)
    return rgbImg


def drawPolygon(rgbImg, points, color, thickness=1):
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(rgbImg, [pts], True, color, thickness)
    return rgbImg


def drawMakeUp(rgbImg, reps, persons, confidences, face_landmarks):
    for r, person, confidence, face_landmark in zip(reps, persons, confidences, face_landmarks):

        # Make the eyebrows into a nightmare
        drawPolygon(rgbImg, face_landmark['left_eyebrow'], (68, 54, 39, 128))
        drawPolygon(rgbImg, face_landmark['right_eyebrow'], (68, 54, 39, 128))
        drawContinuousLines(rgbImg, face_landmark['left_eyebrow'], (68, 54, 39, 150), 5)
        drawContinuousLines(rgbImg, face_landmark['right_eyebrow'], (68, 54, 39, 150), 5)

        # Gloss the lips
        drawPolygon(rgbImg, face_landmark['top_lip'], (150, 0, 0, 128))
        drawPolygon(rgbImg, face_landmark['bottom_lip'], (150, 0, 0, 128))
        drawContinuousLines(rgbImg, face_landmark['top_lip'], (150, 0, 0, 64), 8)
        drawContinuousLines(rgbImg, face_landmark['bottom_lip'], (150, 0, 0, 64), 8)

        # Sparkle the eyes
        drawPolygon(rgbImg, face_landmark['left_eye'], (255, 255, 255, 30))
        drawPolygon(rgbImg, face_landmark['right_eye'], (255, 255, 255, 30))

        # Apply some eyeliner
        drawContinuousLines(rgbImg, face_landmark['left_eye'] + [face_landmark['left_eye'][0]], (0, 0, 0, 110), 6)
        drawContinuousLines(rgbImg, face_landmark['right_eye'] + [face_landmark['right_eye'][0]], (0, 0, 0, 110), 6)

        #  draw text
        bb = r[0]
        text = '{}-{:.2f}'.format(person, confidence)
        rgbImg = drawText(rgbImg, text, bb.left(), bb.top())
    return rgbImg

