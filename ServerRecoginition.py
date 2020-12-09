
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import tkinter as tk
from tkinter import filedialog

import os
import sys
import numpy as np
import dlib
from PIL import Image, ImageFont, ImageDraw
import json
# import jsons
import cv2
import requests

root = tk.Tk()
root.withdraw()

app = Flask(__name__)
CORS(app)

image_index = 0
imageEncode_folder = "imageEncodes"
imageRecoginition_folder = "Recoginition"

face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
pose_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

@app.route('/images/save', methods=['POST'])
def save_image():
    global image_index
    image_index += 1

    # define options for opening
    options = {}
    options['defaultextension'] = ".jpg"
    options['initialfile'] = str(image_index)

    image_data = request.json['image_data']
    image_data = image_data[22:]                   # remove header padding

    f=filedialog.asksaveasfile(mode='wb', **options)
    if f:
        img_path = f.name
        image_index = int(os.path.basename(img_path).split(".jpg")[0])
        f.write(base64.b64decode(image_data))
        f.close
        get_image_encoder(img_path)
    return jsonify('OK')

@app.route('/images/recognize', methods=['POST'])
def recognize():
    image_data = request.json['image_data']
    image_data = image_data[22:]  # remove header padding

    img_path = os.path.join(imageRecoginition_folder, "recoginitionServer.jpg")
    # print(img_path)
    f = open(img_path, 'wb')
    f.write(base64.b64decode(image_data))
    f.close()

    img = np.array(Image.open(img_path))
    emb = get_face_encode(img)
    if emb is not None:
        compare_image_encode(emb)
        draw_face_location(img_path)
    else:
        print("Can't detect face")

    return jsonify('OK')

@app.route('/images/SaveRecognize', methods=['POST'])
def SaveRecognize():
    image_data = request.json['image_data']
    image_data = image_data[22:]                   # remove header padding

    # define options for opening
    options = {}
    options['defaultextension'] = ".jpg"
    f=filedialog.asksaveasfile(mode='wb', **options)
    if f:
        img_path = f.name
        f.write(base64.b64decode(image_data))
        f.close
        img1 = np.array(Image.open(img_path))
        emb = get_face_encode(img1)
        if emb is not None:
            compare_image_encode(emb)
            draw_face_location(img_path)
        else:
            print("Can't detect face")
    return jsonify('OK')

def get_face_location(img):
    locations = face_detector(img)
    return locations[0] if len(locations) > 0 else None
def get_face_landmark(img):
    face_location = get_face_location(img)
    return pose_predictor(img, face_location) if face_location else None
def get_face_encode(img):
    landmark = get_face_landmark(img)
    return np.array(face_encoder.compute_face_descriptor(img, landmark)) if landmark else None
def get_emb_distance(emb1, emb2):
    if (emb1 is None) or (emb2 is None):
        return 1.0
    return np.sqrt(np.sum((emb1 - emb2) ** 2))

def get_image_encoder(img_path):
    path = os.path.dirname(img_path)
    folder_name = os.path.basename(path)
    file_name = os.path.basename(img_path).split(".jpg")[0]
    imageEncode_folder_path = os.path.join(imageEncode_folder, folder_name)
    if not os.path.exists(imageEncode_folder_path):
        os.makedirs(imageEncode_folder_path)
    imgEncode_path = os.path.join(imageEncode_folder_path, file_name + ".dat")
    # print(imgEncode_path)

    img = np.array(Image.open(img_path))
    emb = get_face_encode(img)
    if emb is not None:
        emb.tofile(imgEncode_path)
        print("Save emb face")
    else:
        print("Can't detect face")

def compare_distance(emb1, emb2, j):
    global num, name
    distance = get_emb_distance(emb1, emb2)
    if distance < 0.5:
        # print(distance)
        num += 1
        name = j

def compare_image_encode(emb1):
    global num, name
    for j in os.listdir(imageEncode_folder):
        num = 0
        imageEncode_folder_path = os.path.join(imageEncode_folder, j)
        for f in os.listdir(imageEncode_folder_path):
            imageEncode_path = os.path.join(imageEncode_folder_path, f)
            emb2 = np.fromfile(imageEncode_path, dtype=float)
            compare_distance(emb1, emb2, j)
        determine_persion()

def draw_face_location(img_path):
    global name
    # use a truetype font
    font = ImageFont.truetype("arial.ttf", 15)
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    face_location = get_face_location(np.array(img))
    # print(face_location)
    rec = [face_location.left(), face_location.top(), face_location.right(), face_location.bottom()]
    draw.rectangle(rec, fill=None)
    draw.text((face_location.left() + 40, face_location.top() - 30), name, (255, 255, 0), font=font)
    ImageDraw.Draw(img)
    img.save("Recoginition/Recoginition.jpg")
    name = ""

def determine_persion():
    # print("num = ", num)
    if num > 2:
        print("Person Name: ", name, "(", num, ")")


@app.route('/', methods=['GET'])
def home():
    return send_from_directory(".", "takeImage.html")

@app.route('/getImage', methods=['GET'])
def ImageServer():
    img_path = "Recoginition/Recoginition.jpg"
    with open(img_path, "rb") as imageFile:
        img_data = base64.b64encode(imageFile.read())
    img_data_json = "data:image/jpeg;base64," + img_data.decode('ascii')
    return img_data_json

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=False)