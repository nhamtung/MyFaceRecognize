

import os
import sys
import numpy as np
import dlib
from PIL import Image

imageCapture_folder = "imageCapture"
imagesReconigition_folder = "Recoginition"
imageEncode_folder = "imageEncodes"

face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
pose_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

num = 'global'
name = "global"

def get_face_location(img):
    locations = face_detector(img)
    return locations[0] if len(locations) > 0 else None

def get_face_landmark(img):
    face_location = get_face_location(img)
    # print("face_location: ", face_location)
    return pose_predictor(img, face_location) if face_location else None

def get_face_encode(img):
    landmark = get_face_landmark(img)
    # print("landmark: ", landmark)
    return np.array(face_encoder.compute_face_descriptor(img, landmark)) if landmark else None

def get_emb_distance(emb1, emb2):
    if (emb1 is None) or (emb2 is None):
        return 1.0
    return np.sqrt(np.sum((emb1 - emb2) ** 2))

def compare_distance(emb1, emb2, j):
    global num, name
    distance = get_emb_distance(emb1, emb2)
    # print(distance)
    if distance < 0.4:
        num += 1
        name = j

def read_image_encode(emb1):
    global num, name
    name_ = "Who are you?"
    for j in os.listdir(imageEncode_folder):
        num = 0
        imageEncode_folder_path = os.path.join(imageEncode_folder, j)
        for f in os.listdir(imageEncode_folder_path):
            imageEncode_path = os.path.join(imageEncode_folder_path, f)
            emb2 = np.fromfile(imageEncode_path, dtype=float)
            compare_distance(emb1, emb2, j)
        if num > 0:
            name_ = name
    return name_


def faceEncode(nameFolderPersion):
    imageCapture_folder_path = os.path.join(imageCapture_folder, nameFolderPersion)
    imageEncode_folder_path = os.path.join(imageEncode_folder, nameFolderPersion)
    i = 0
    for f in os.listdir(imageCapture_folder_path):
        i = i+1
        img_path = os.path.join(imageCapture_folder_path, f)
        img = np.array(Image.open(img_path))
        emb = get_face_encode(img)

        if not os.path.exists(imageEncode_folder_path):
            os.makedirs(imageEncode_folder_path)
        imgEncode_path = os.path.join(imageEncode_folder_path, str(i)+".dat")
        print(imgEncode_path)
        emb.tofile(imgEncode_path)