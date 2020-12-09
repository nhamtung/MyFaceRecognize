

import os
import sys
import numpy as np
import dlib
from PIL import Image

face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
pose_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

def get_face_location(img):
    locations = face_detector(img)
    return locations[0] if len(locations) > 0 else None

def get_face_landmark(img):
    face_location = get_face_location(img)
    return pose_predictor(img, face_location) if face_location else None

def get_face_encode(img):
    landmark = get_face_landmark(img)
    return np.array(face_encoder.compute_face_descriptor(img, landmark)) if landmark else None


def get_emb_distance(img1, img2):
    emb1 = get_face_encode(img1)
    emb2 = get_face_encode(img2)
    if (emb1 is None) or (emb2 is None):
        return 1.0
    return np.sqrt(np.sum((emb1 - emb2) ** 2))


img1 = np.array(Image.open(sys.argv[1]))
img2 = np.array(Image.open(sys.argv[2]))
print(get_emb_distance(img1, img2))