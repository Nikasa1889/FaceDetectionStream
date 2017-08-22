import os
import time
import cv2
import face_recognition
import dlib
# This FaceDetection class use dlib for face detection, face encoding, and face comparision
UP_SAMPLE = 1
MESSAGE_FILE = "./welcomeMessages.txt"
fileDir = os.path.dirname(os.path.realpath(__file__))
faceDir = os.path.join(fileDir,'images', 'faces')
valid_images = [".jpg", ".png", ".jpeg"]

class FaceDetection():
    def __init__(self,
                 faceDir=faceDir):
        self.listOfKnownFaceEncodings = []
        self.listOfKnownFaceNames = []
        print("Face dir: ", faceDir)
        for fname in os.listdir(faceDir):
            (name, ext) = os.path.splitext(fname)
            print(fname, " ", ext)
            if ext.lower() not in valid_images:
                continue
            img = face_recognition.load_image_file(os.path.join(faceDir, fname))
            face_encodings, _ = face_recognition.face_encodings(img)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                self.listOfKnownFaceEncodings.append(face_encoding)
                self.listOfKnownFaceNames.append(name.split("_")[0])
        # Check consistency between welcome messages and faceExamples
        self.welcomeMessages = []
        with open(MESSAGE_FILE, "r") as f:
            self.welcomeMessages = eval(f.read().encode("utf-8"))
        nameWelcome = set(self.welcomeMessages.keys())
        nameFace = set(self.listOfKnownFaceNames)
        if (nameWelcome.issubset(nameFace)
            and nameFace.issubset(nameWelcome)):
            print("Known name: ", self.listOfKnownFaceNames)
        else:
            print("Names in welcomeMessage but not face examples: {}".format(nameWelcome.difference(nameFace)))
            print("Names in face examples but not in welcomeMessage: {}".format(nameFace.difference(nameWelcome)))
            raise ValueError('Names in the face example images are not consistent with welcomeMessage file')

    def infer(self, rgbImg, multiple=True):
        start = time.time()
        face_locations = face_recognition.face_locations(rgbImg, number_of_times_to_upsample=UP_SAMPLE)
        print("Face detection took {} seconds.".format(time.time() - start))
        start = time.time()
        face_encodings, raw_landmarks = face_recognition.face_encodings(rgbImg, face_locations)
        print("Face encoding took {} seconds.".format(time.time() - start))
        reps = []
        persons = []
        confidences = []
        for (face_encoding, face_location) in zip(face_encodings, face_locations):
            bb = dlib.rectangle(top=face_location[0],
                                right=face_location[1],
                                bottom=face_location[2],
                                left=face_location[3])
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
        return reps, persons, confidences, raw_landmarks

    def recognize_face_landmark(self, rgbImg, raw_landmarks):
        return face_recognition.face_landmarks(rgbImg, None, raw_landmarks)
