from __future__ import print_function
from detect.MtcnnDetector_rw import MtcnnDetector
from detect.detector_rw import Detector
from detect.fcn import FcnDetector
from detect.mtcnn_model_rw import P_Net, R_Net, O_Net
from recognize.facenet import FaceNet
import cv2
import numpy as np
import os

# Init MtcnnDetector
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = []
prefix = ['detect/MTCNN_model/PNet_landmark/PNet', 'detect/MTCNN_model/RNet_landmark/RNet', 'detect/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

detectors.append(FcnDetector(P_Net, model_path[0]))
detectors.append(Detector(R_Net, 24, batch_size[1], model_path[1]))
detectors.append(Detector(O_Net, 48, batch_size[2], model_path[2]))

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

# Init FaceNet
model_path = 'recognize/facenet_model'
face_net = FaceNet(model_path)

test_path = os.path.join(os.getcwd(), '..', 'CNN_FacePoint/test/bioid_jpg')
img1 = cv2.imread(os.path.join(test_path, '0001.jpg'))
img2 = cv2.imread(os.path.join(test_path, '0006.jpg'))

faces1 = mtcnn_detector.get_face_from_single_image(img1)#detections
faces2 = mtcnn_detector.get_face_from_single_image(img2)

print("faces detected from original image:", len(faces1), len(faces2))

boxes1,landmarks1 = mtcnn_detector.detect(img1)
boxes2,landmarks2 = mtcnn_detector.detect(img2)

print("boxes detected from aligned image:", len(boxes1), len(boxes2))
print("landmarks detected from aliened image:", len(landmarks1), len(landmarks2))
print("boxes1", boxes1)
print("boxes2", boxes2)
print("landmarks1", landmarks1)
print("landmarks2", landmarks2)

print("difference between bbox", (np.subtract(boxes1[:,:4],boxes2[:,:4])))
print("mse between bbox", np.sqrt(np.sum(np.square(np.subtract(boxes1[:,:4],boxes2[:,:4])))))
print("mse between landmarks", np.sqrt(np.sum(np.square(np.subtract(landmarks1,landmarks2)))))
#cls,reg,landmark = mtcnn_detector.detect_onet(faces1[0], faces2[0])


emb1 = face_net.predict(faces1[0])
emb2 = face_net.predict(faces2[0])
#print("emb1 with facenet: ", emb1)
#print("emb2 with facenet: ", emb2)
print("score: ")
print(np.sqrt(np.sum(np.square(np.subtract(emb1, emb2)))))
