

import os
import sys
import numpy as np
import dlib
from PIL import Image

images_folder = "images"
imageEncode_folder = "imageEncodes"

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

images_folder_path = os.path.join(images_folder, sys.argv[1])
imageEncode_folder_path = os.path.join(imageEncode_folder, sys.argv[1])
i = 0
for f in os.listdir(images_folder_path):
    i = i+1
    img_path = os.path.join(images_folder_path, f)
    img = np.array(Image.open(img_path))
    emb = get_face_encode(img)

    if not os.path.exists(imageEncode_folder_path):
        os.makedirs(imageEncode_folder_path)
    imgEncode_path = os.path.join(imageEncode_folder_path, str(i)+".dat")
    print(imgEncode_path)
    emb.tofile(imgEncode_path)