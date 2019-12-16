import numpy as np
import face_recognition
import sys
import os
import cv2
import pickle
from sklearn import neighbors
import imutils
import math
import modules.common_params as g
import modules.face_train as train
# Class to handle face recognition

class Face:

    def __init__(self, upsample_times=1, num_jitters=0, model='cnn'):
        g.log.debug('Initializing face recognition with model:{} upsample:{}, jitters:{}'
                       .format(model, upsample_times, num_jitters))

        self.upsample_times = upsample_times
        self.num_jitters = num_jitters
        self.model = model
        self.knn = None

        encoding_file_name = g.config['known_faces_path']+'/faces.dat'
        # to increase performance, read encodings from  file
        if (os.path.isfile(encoding_file_name)):
            g.log.debug ('pre-trained faces found, using that. If you want to add new images, remove: {}'.format(encoding_file_name))
          
            #self.known_face_encodings = data["encodings"]
            #self.known_face_names = data["names"]
        else:
            # no encodings, we have to read and train
            g.log.debug ('trained file not found, reading from images and doing training...')
            
            
            train.train()
        with open(encoding_file_name, 'rb') as f:
            self.knn  = pickle.load(f)


    def get_classes(self):
        return self.knn.classes_

    def _rescale_rects(self, a):
        rects = []
        for (left, top, right, bottom) in a:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            rects.append([left, top, right, bottom])
        return rects

    def detect(self, image):
        labels = []
        classes = []
        conf = []

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = image[:, :, ::-1]
        #rgb_image = image

        # Find all the faces and face encodings in the target image
        face_locations = face_recognition.face_locations(rgb_image, model=self.model, number_of_times_to_upsample=self.upsample_times)
        face_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations, num_jitters=self.num_jitters)
        
        if not len(face_encodings):
            g.log.debug ('Face recognition: no faces found')
            return [],[],[]

        # Use the KNN model to find the best matches for the test face
        closest_distances = self.knn.kneighbors(face_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= g.config['face_recog_dist_threshold'] for i in range(len(face_locations))]

        matched_face_names = []
        matched_face_rects = []

        for pred, loc, rec in zip(self.knn.predict(face_encodings), face_locations, are_matches):
            label = pred if rec else g.config['unknown_face_name']
            matched_face_rects.append((loc[3], loc[0], loc[1], loc[2]))
            matched_face_names.append(label)
            conf.append(1)

        
        detections = []

        for l, c, b in zip(matched_face_names, conf, matched_face_rects):
            c = "{:.2f}%".format(c * 100)
            obj = {
                'type': l,
                'confidence': c,
                'box': b
            }
            detections.append(obj)

        return detections

       




