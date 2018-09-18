# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 02:14:55 2018

@author: rein9
"""

from __future__ import print_function
import cv2
import numpy as np
import os
import tensorflow as tf

from flask import (
    Flask,
    request,
    render_template,
    jsonify
)
from flask_bootstrap import Bootstrap

from detect.MtcnnDetector_rw import MtcnnDetector
from detect.detector_rw import Detector
from detect.fcn import FcnDetector
from detect.mtcnn_model_rw import P_Net, R_Net, O_Net
from detect_acc import detect_face
from recognize.facenet import FaceNet

# Initialize MtcnnDetector
threshold = [0.9,0.6,0.7]
min_face_size = 24
stride = 2
batch_size = [2048,256,16]
prefix = ['detect/MTCNN_model/PNet_landmark/PNet', 'detect/MTCNN_model/RNet_landmark/RNet', 'detect/MTCNN_model/ONet_landmark/ONet']
epoch = [18,14,16]
sw =False
detectors = []
shuffle = False
model_path = ['%s-%s' % (x, y) for (x, y) in zip(prefix, epoch)]

# =============================================================================
# In the old way, we use facenet to predict
# =============================================================================
detectors.append(FcnDetector(P_Net, model_path[0]))
detectors.append(Detector(R_Net, 24, batch_size[1], model_path[1]))
detectors.append(Detector(O_Net, 48, batch_size[2], model_path[2]))#shouldnt it be O_Net?

test_mode = 'ONet'
# load pnet
if sw:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
#load rnet
if test_mode in ['RNet', 'ONet']:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet
#load onet
if test_mode is 'ONet':
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=threshold, slide_window=sw)

# Read the existing ready trained MtcnnModel for now
# model_path = 'recognize/facenet_model'
# face_net = FaceNet(model_path)
webface =  Flask(__name__)
bootstrap = Bootstrap(webface)
@webface.route('/')
#this is a mask when '/' is detected will call the following
def index():
    return render_template('index.html')

def calc_score(img1, img2):
    threshold = 0.35
    face1= mtcnn_detector.get_face_from_single_image(img1)
    face2= mtcnn_detector.get_face_from_single_image(img2)
    print(face1)
    print(face2)
    if len(face1) != 1 or len(face2) != 1:
        '''
        check if there is only one face in the image
        '''
        return 'Please upload image with exact one person.', 0
# =============================================================================
# Old way, loading pretrained resnet1
#     emb1 = face_net.predict(face1[0])
#     emb2 = face_net.predict(face2[0])
# =============================================================================
    emb1 = detectors[2].predict(face1[0])
    emb2 = detectors[2].predict(face2[0])

    score = np.sqrt(np.sum(np.square(np.substract(emb1, emb2))))
    if score < threshold:
        return 'Same person with score %s' % str(score)
    else:
        return 'Not the same person with score %s' % str(score)

@webface.route('/get_score', methods=['POST'])
def get_score():
    '''
    '''
    if request.method == 'POST':
        files1 = request.files['file1']
        files2 = request.files['file2']
        img1 = cv2.imdecode(np.fromstring(files1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.fromstring(files2.read(), np.uint8), cv2.IMREAD_COLOR)

        result, score = calc_score(img1, img2)
        return jsonify(result=result, score=score)

if __name__ == '__main__':
    webface.run('0.0.0.0')