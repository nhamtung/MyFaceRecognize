from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import dlib
import io
import os
import json
import math

import base64
app = Flask(__name__)
CORS(app)
data_dir = "data"
user_list_path = "user.json"

EMBED_SIZE = 128
EMBED_DISTANCE_THRESH = 0.35

face_detector = dlib.get_frontal_face_detector()
pose_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def getRectArea(rect):
    w = rect.right() - rect.left()
    h = rect.bottom() - rect.top()
    return w * h

def getFaceLocation(img):
    locations = face_detector(img)
    if len(locations) > 0:
        locations = sorted(locations, key=getRectArea, reverse=True)
        return locations[0]
    return None    
    
def getFaceEncode(img):
    face_location = getFaceLocation(img)
    if face_location is not None:
        face_land_mark = pose_predictor(img, face_location)
        return np.array(face_encoder.compute_face_descriptor(img, face_land_mark), dtype=np.float32)
    return None
    
def loadUserList():
    user_list = []
    if os.path.exists(user_list_path):
        f = open(user_list_path)
        user_list = json.loads(f.read())
        f.close()
    return user_list
    
def saveUserList(user_list):
    f = open(user_list_path, 'w')
    f.write(json.dumps(user_list))
    f.close()    

def addNewImage(username, image_data):
    bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(bytes))
    img = np.array(img)
    face_encode = getFaceEncode(img)
    if face_encode is not None:
        face_encode = np.array([face_encode])
        user_list = loadUserList()
        usernames = [user['name'] for user in user_list]
        user_ids = [user['id'] for user in user_list]
                    
        if username in usernames:
            index = usernames.index(username)
            user_id = user_ids[index]
            data_file = os.path.join(data_dir, '{}.np'.format(user_id))
            data = np.fromfile(data_file, dtype=np.float32)
            data = data.reshape(-1, EMBED_SIZE)
            data = np.concatenate((data, face_encode), axis=0)
            data.tofile(data_file)
        else:
            max_user_id = max(user_ids) if len(user_ids) > 0 else 0
            user_id = max_user_id + 1
            user_list.append({'name' : username, 'id' : user_id})
            saveUserList(user_list)
            
            data_file = os.path.join(data_dir, '{}.np'.format(user_id))
            face_encode.tofile(data_file)   
    
def findClosestUser(image_data):
    bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(bytes)) 
    img = np.array(img)    
    face_encode = getFaceEncode(img)
    if face_encode is not None:
        face_encode = np.array([face_encode])
        user_list = loadUserList()
        usernames = [user['name'] for user in user_list]
        user_ids = [user['id'] for user in user_list]
        
        distances = []
        for i, user_id in enumerate(user_ids):
            data_file = os.path.join(data_dir, '{}.np'.format(user_id))
            data = np.fromfile(data_file, dtype=np.float32)
            data = data.reshape(-1, EMBED_SIZE)
            distance2 = np.min(np.sum((data - face_encode)**2, axis=1))
            distance = math.sqrt(distance2)
            distances.append((distance, i))
            
        if len(distances) > 0:
            distances = sorted(distances)
            dist, index = distances[0]
            print(dist, index)
            if dist <= EMBED_DISTANCE_THRESH:
                return usernames[index]
    return ""
    
@app.route('/')
def mainPage():
    return send_from_directory('.', 'index.html')

@app.route('/add_user_image', methods=['POST'])
def addUserImage():
    print(request.json)
    image_data = request.json['image_data']
    username = request.json['username']    
    image_data = image_data[23:]
    addNewImage(username, image_data)    
    return jsonify('OK')

@app.route('/recognize', methods=['POST'])
def recognize():
    image_data = request.json['image_data']
    image_data = image_data[23:]
    username = findClosestUser(image_data)    
    return jsonify({'username': username})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', ssl_context=('cert.pem', 'key.pem'))
