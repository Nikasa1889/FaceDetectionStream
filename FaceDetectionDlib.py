import os
import time
import cv2
import face_recognition
import dlib
# This FaceDetection class use dlib for face detection, face encoding, and face comparision
UP_SAMPLE = 1
fileDir = os.path.dirname(os.path.realpath(__file__))
faceDir = os.path.join(fileDir,'faces', 'faceExamples')
valid_images = [".jpg", ".png", ".jpeg"]
class FaceDetection():
    def __init__(self, 
            faceDir=faceDir):
        #TODO: Load face examples of each person here. Each person can have multiple faces
        self.listOfKnownFaceEncodings = []
        self.listOfKnownFaceNames = []
        print("Face dir: ", faceDir)
        for fname in os.listdir(faceDir):
            (name, ext) = os.path.splitext(fname)
            print (fname, " ", ext)
            if ext.lower() not in valid_images:
                continue
            img = face_recognition.load_image_file(os.path.join(faceDir, fname))
            face_encodings = face_recognition.face_encodings(img)
            if len(face_encodings)>0:
                face_encoding = face_encodings[0]
                self.listOfKnownFaceEncodings.append(face_encoding)
                self.listOfKnownFaceNames.append(name.split("_")[0])

        print("Known name: ", self.listOfKnownFaceNames);
    def infer(self, rgbImg, multiple=True):
        start = time.time()
        face_locations = face_recognition.face_locations(rgbImg, number_of_times_to_upsample = UP_SAMPLE)
        print("Face detection took {} seconds.".format(time.time()-start))
        start = time.time()
        face_encodings = face_recognition.face_encodings(rgbImg, face_locations)
        print("Face encoding took {} seconds.".format(time.time()-start))

        reps = []
        persons = []
        confidences = []
        for (face_encoding, face_location) in zip(face_encodings, face_locations):
            bb = dlib.rectangle(top = face_location[0], 
                                right = face_location[1],
                                bottom = face_location[2],
                                left = face_location[3])
            reps.append((bb, face_encoding))
            distances = list(face_recognition.face_distance(self.listOfKnownFaceEncodings, face_encoding))
            if (min(distances) < 0.6):
                idxMinDistance = distances.index(min(distances))
                person = self.listOfKnownFaceNames[idxMinDistance]
                confidence = 0.6 - min(distances)
            else:
                person = "Unknown"
                confidence = 0
            persons.append(person)
            confidences.append(confidence)
        return (reps, persons, confidences)
