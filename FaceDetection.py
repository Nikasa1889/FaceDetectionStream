import cv2
import pickle
import sys
import os
import numpy as np
import pandas as pd
import openface
import time
import dlib

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..','..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

class FaceDetection():

    def __init__(self, 
            classifierModel="./trainingData/classifier.pkl",
            dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat"),
            networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7'),
            imgDim = 96,
            cuda = True,
            verbose = True,
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.5,
            fontThickness = 1,
            textBaseline = 0):
        self.classifierModel = classifierModel
        self.dlibFacePredictor = dlibFacePredictor
        self.networkModel = networkModel

        self.imgDim = imgDim
        self.cuda = cuda
        self.verbose = verbose
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.fontThickness = fontThickness
        self.textBaseline = textBaseline
        

        self.align = openface.AlignDlib(self.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.networkModel, imgDim=self.imgDim,
                                                  cuda=self.cuda)


    def getFaceBBs(self, rgbImg, multiple=True):
        start = time.time()
        align = self.align
        if multiple:
            bbs = align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]
        if self.verbose:
            print("Face detection took {} seconds.".format(time.time() - start))
        return bbs

    def getReps(self, rgbImg, bbs, multiple=True):
        net = self.net
        align = self.align
        if len(bbs) == 0 or (not multiple and bb1 is None):
            print("No face")
            return []
        else:
            reps = []
            for bb in bbs:
                start = time.time()
                alignedFace = align.align( self.imgDim, rgbImg, bb,
                            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                if alignedFace is None:
                    print("Unable to align a face image")
                    continue;
                if self.verbose:
                    print("Alignment took {} seconds.".format(time.time() - start))
                    print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))
        
                start = time.time()
                rep = net.forward(alignedFace)
                if self.verbose:
                    print("Neural network forward pass took {} seconds.".format(
                        time.time() - start))
                reps.append((bb, rep))
            sreps = sorted(reps, key=lambda x: x[0])
            return sreps

    def drawBox(self, rgbImg, bb, person, confidence):
        #Draw bounding box first
        cv2.rectangle(rgbImg, 
                (bb.left(), bb.bottom()), (bb.right(), bb.top()),
                (127, 255, 212),
                4)
        #Calculate text length
        textSize = cv2.getTextSize(
                '{}-{:.2f}'.format(person.decode('utf-8'), confidence), 
                    self.fontFace, self.fontScale, self.fontThickness)[0]
        #Draw the text background
        cv2.rectangle(rgbImg, 
                (bb.left(), bb.top()), 
                (bb.left() + textSize[0], bb.top() -textSize[1]), 
                (127, 255, 212), -1);
        #Now put the text on it
        cv2.putText(rgbImg, 
                '{}-{:.2f}'.format(person.decode('utf-8'), confidence), 
                (bb.left(), bb.top()-1),
                self.fontFace, self.fontScale,(0, 0, 0), self.fontThickness
                )
        return rgbImg

    def drawBoxes (self, rgbImg, reps, persons, confidences):
        for r, person, confidence in zip(reps, persons, confidences):
            bb = r[0]
            rgbImg = self.drawBox(rgbImg, bb, person, confidence)
        return rgbImg

    def infer(self, rgbImg, multiple=True):
        with open(self.classifierModel, 'rb') as f:
            if sys.version_info[0] < 3:
                    (le, clf) = pickle.load(f)
            else:
                    (le, clf) = pickle.load(f, encoding='latin1')
    
        print("====================================")

        bbs = self.getFaceBBs(rgbImg, multiple)
        reps = self.getReps(rgbImg, bbs, multiple)
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
                print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'), bbx,
                                                                             confidence))
            else:
                print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
            
        #return (self.drawBoxes(rgbImg, reps, persons, confidences))
        return (reps, persons, confidences)
