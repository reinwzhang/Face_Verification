# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 03:33:24 2018
last update on Aug 22nd on surfacebook

@author: rein9
"""

from __future__ import print_function
from __future__ import absolute_import
import cv2
import time
import numpy as np
from .nms import py_nms
from .MTCNN_config import config

class MtcnnDetector(object):
    def __init__(self,
                 detectors,
                 min_face_size=25,
                 stride=2,
                 threshold=[0.6,0.7,0.7],
                 scale_factor=0.79,
                 slide_window=False):
        self.pnet_detector = detectors[0] #fcn
        self.rnet_detector = detectors[1] #detectors
        self.onet_detector = detectors[2] #detectors
        self.min_face_size = min_face_size
        self.stride = stride
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window

    def convert_to_square(self, bbox):
        '''
        this is to convert an bbox to square image
        Parameters:
            bbox: numpy array , shape: n x 5
            input: bbox
        Returns:
            square bbox
        '''
        bbox_copy = bbox.copy()
        h = bbox[:,3] - bbox[:,1] +1
        w = bbox[:,2] - bbox[:,0] +1
        max_size = np.maximum(h, w)
        bbox_copy[:,1] = bbox[:,1] - (max_size - h) * 0.5
        bbox_copy[:,0] = bbox[:,0] - (max_size - w) * 0.5
        bbox_copy[:,3] = bbox_copy[:,1] + max_size -1
        bbox_copy[:,2] = bbox_copy[:,0] + max_size -1
        return bbox_copy

    def calibrate_box(self, bbox, reg):
        '''
        calibrate bboxes
        Parameters:
        bbox: numpy array, shape n x 5
          input bboxes
        reg:  numpy array, shape n x 4
          bboxes adjustment
        Returns:
          bboxes after refinement
        '''
        bbox_c = bbox.copy()
        # add one dimension for w & h
        w = bbox[:,2] - bbox[:,0] +1
        w = np.expand_dims(w,1)
        h = bbox[:,3] - bbox[:,1] +1
        h = np.expand_dims(h,1)
        reg_m = np.hstack([w,h,w,h]) #shape: n*4
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] +aug
        return bbox_c

    def processed_image(self, image, scale):
        '''
        To scale the image
        '''
        height, width, channels = image.shape
        scaled_h = int(height*scale)
        scaled_w = int(width*scale)
        scaled_dim = (scaled_w, scaled_h)
        img_resized = cv2.resize(image, scaled_dim, interpolation=cv2.INTER_LINEAR)
        #normalize
        img_resized = (img_resized - 127.5)/128
        return img_resized

    def generate_bbox(self, cls_map, reg, scale, threshold):
        '''
        generate bbox from feature cls_map
        paramters:
          cls_map: numpy.array, n*m
          reg:numpy.array, n*m*4
          scale: float number, scale of this detection
          threshold: float number
          returns:
            bbox array
        '''
        #PNet
        stride=2
        cellsize=12
#        print("the cls_map is: ", cls_map)
        t_index = np.where(cls_map > threshold)
        if t_index[0].size == 0:
            return np.array([])
        #get the offset
        dx1,dy1,dx2,dy2=[reg[t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1,dy1,dx2,dy2])
        score = cls_map[t_index[0],t_index[1]]#detector prediction
        # divide by scale with project the image to the old imag
        boundingbox = np.vstack([np.round((stride*t_index[1])/scale),
                                 np.round((stride*t_index[0])/scale),
                                 np.round((stride*t_index[1]+cellsize)/scale),
                                 np.round((stride*t_index[0]+cellsize)/scale),
                                 score,
                                 reg])
        return boundingbox.T

    def detect_pnet(self, img):
        """Get face candidates through pnet
        Parameters:
        im: [numpy array] input image array
        Returns:
        boxes: [numpy array] detected boxes before calibration
        boxes_c: [numpy array]boxes after calibration
        """
        h,w,c=img.shape
        net_size =12
        image_scale = float(net_size)/self.min_face_size
        img_resized= self.processed_image(img, image_scale)
        resized_h, resized_w, _ = img_resized.shape
        all_boxes = list()
        while min(resized_h, resized_w) > net_size:
            #recursively scaling until the image reaches minimun size
            print("PNet resized image", resized_h, resized_w)
            cls_map, reg = self.pnet_detector.predict(img_resized) #reg : bbox_pred from pnet
#            print("PNET cls_map and reg", cls_map.shape, reg.shape,"net_size", net_size)
            boxes = self.generate_bbox(cls_map[:,:,1], reg, image_scale, self.threshold[0])
            image_scale *= self.scale_factor
            # very important, always scale from the original image
            img_resized = self.processed_image(img, image_scale)
            resized_h, resized_w, _ = img_resized.shape

            if boxes.size==0:
                continue
            #otherwise get the index of the bbox with nms, we need to manunally adjust the threshold
            keep = py_nms(boxes[:,:5],0.5,mode='Union') #to get the index of boxes to keep with regard to nms < 0.5
            boxes = boxes[keep]
#            print("box shape after nms", boxes.shape)
            all_boxes.append(boxes)
#        print("After image pramid, the cls are", cls_map)
#        print("After image pramid, the reg are", reg)
        if len(all_boxes) == 0:
            return None,None,None
        all_boxes = np.vstack(all_boxes)
        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = np.vstack(all_boxes)
        #adjust the bounding box
        boxes = all_boxes[:, :5]
        bbw = all_boxes[:,2]-all_boxes[:,0]+1
        bbh = all_boxes[:,3]-all_boxes[:,1]+1

        #refine the boxes
        boxes_c = np.vstack([all_boxes[:,0] + all_boxes[:,5]*bbw,
                            all_boxes[:,1] + all_boxes[:,6]*bbh,
                            all_boxes[:,2] + all_boxes[:,7]*bbw,
                            all_boxes[:,3] + all_boxes[:,8]*bbh,
                            all_boxes[:,4]])
        boxes_c = boxes_c.T
        return boxes,boxes_c,None

    def detect_rnet(self, img, dets):
        """Get face candidates using rnet

        Parameters:
        im: [numpy array] input image array
        dets: [numpy array] detection results of pnet
        Returns:
        boxes: [numpy array] detected boxes before calibration
        boxes_c:[numpy array] boxes after calibration
        """
        h,w,c = img.shape
        net_size = 24
        dets = self.convert_to_square(dets)# if rectangular, convert to the closest square
        dets[:, 0:4] = np.round(dets[:, 0:4])
        #return of pad  [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]= self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype = np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype = np.uint8)# why uint8?
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :] # this is to normalize the image
            cropped_ims[i,:,:,:] = (cv2.resize(tmp,(24,24))-127.5)/128
        #cls_scores : num_data*2
        #reg: num_data*4
        #landmark: num_data*10
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]# copy all scores where there is image
        keep_inds = np.where(cls_scores>self.threshold[1])[0]
#        print("indexs to keep are", keep_inds)
        if len(keep_inds)>0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None, None
        #calibrate boxes_c
        keep = py_nms(boxes, 0.6)# why o.6?
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes,boxes_c,None

    def detect_onet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        im: [numpy array] input image array
        dets: [numpy array] detection results of rnet
        Returns:
        boxes: [numpy array] detected boxes before calibration
        boxes_c:[numpy array] boxes after calibration
        """
        h,w,c = im.shape
        dets = self.convert_to_square(dets)
        dets[:,0:4] = np.round(dets[:,0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype = np.float32)
        for i in range(num_boxes):
            #1. initial
            tmp = np.zeros((tmph[i], tmpw[i],3),dtype= np.uint8)
            #2. crop it
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            #3. normailize it
            cropped_ims[i,:,:,:] = (cv2.resize(tmp, (48, 48)) -127.5)/128
        cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:,1]  #the probability of face
        keep_inds = np.where(cls_scores > self.threshold[2])[0]
        if len(keep_inds)>0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None
        # fully connect?
        w = boxes[:, 2] - boxes[:, 0]+1
        h = boxes[:, 3] - boxes[:, 1]+1
        #.project landmark to all dimensions?
        # array[0::2] means starting from index 0 and step is 2
        # array[1::2] means starting from index 1 and step is 2
        landmark[:, 0::2]= (np.tile(w, (5,1))*landmark[:,0::2].T + np.tile(boxes[:,0], (5,1))-1).T
        landmark[:, 1::2]= (np.tile(h, (5,1))*landmark[:,1::2].T + np.tile(boxes[:,1], (5,1))-1).T
        boxes_c = self.calibrate_box(boxes, reg)

        keep = py_nms(boxes, 0.6, "Minimum")
        boxes = boxes[keep]# only ONET uses minimum
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes,boxes_c,landmark

    def pad(self, bboxes, w, h):
        """
          pad the bboxes, alse restrict the size of it
        Parameters:
      ----------
          bboxes: numpy array, n x 5;
              input bboxes
          w: float number
              width of the input image
          h: float number
              height of the input image
        Returns :
        ------
          dy, dx : numpy array, n x 1
              start point of the bbox in target image
          edy, edx : numpy array, n x 1
              end point of the bbox in target image
          y, x : numpy array, n x 1
              start point of the bbox in original image
          ex, ex : numpy array, n x 1
              end point of the bbox in original image
          tmph, tmpw: numpy array, n x 1
              height and width of the bbox
        """
        tmpw, tmph = bboxes[:,2] - bboxes[:,0]+1, bboxes[:,3] - bboxes[:,1]+1
        num_box = bboxes.shape[0]
        dx,dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx,edy = tmpw.copy()-1,tmph.copy()-1
        x,y,ex,ey = bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]

        tmp_index = np.where(ex>w-1)
        edx[tmp_index] = tmpw[tmp_index]+w-2-ex[tmp_index]
        ex[tmp_index] = w -1

        tmp_index = np.where(ey>h-1)
        edy[tmp_index] = tmph[tmp_index]+h-2-ey[tmp_index]
        ey[tmp_index] = h -1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]
        return return_list

#use for video
    def detect(self, img):
        """Detect face over image
        """
        boxes = None
        t = time.time()

        #if pnet
        t1 = 0
        if self.pnet_detector:
            boxes, boxes_c,_=self.detect_pnet(img)
            if boxes_c is None:
                # boxes_c represents boxes after calibration
                return np.array([]), np.array([])
            t1 = time.time()-t
            t = time.time()
            print("pnet_ouput boxes: ", boxes.shape)
            print("pnet_ouput boxes_c: ", boxes_c.shape)

        # rnet
        t2 = 0
        if self.rnet_detector:
            boxes, boxes_c,_ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])
            t2 = time.time() - t
            t = time.time()
            print("rnet_ouput boxes: ", boxes.shape)
            print("rnet_ouput boxes_c: ", boxes_c.shape)
        # onet
        t3 = 0
        if self.onet_detector:
            boxes, boxes_c,landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])
            t3 =  time.time()-t
            t = time.time()
            print("onet_ouput boxes: ", boxes.shape)
            print("onet_ouput boxes_c: ", boxes_c.shape)
            print("onet_ouput landmark: ", landmark.shape)
            print('time cost for detection: ' + '{:.3f}'.format(t1+t2+t3)+ 'pnet: (:.3f) rnet: {:.3f} onet: {:.3f}'.format(t1, t2,t3))
        return boxes_c, landmark

    def detect_face(self, test_data):
        #detect face from video stream
        all_boxes = []#same each image
        landmarks = []
        batch_idx = 0
        sum_time = 0
        # test data as iter_
        for databatch in test_data:
#            print("image: ", databatch)
            #databatch(image returned)
            if batch_idx % 100 == 0:#100 as the batch numer
                print('%d images done' % batch_idx)
            im = databatch
            #pnet
            t1 = 0
            if self.pnet_detector:
                t = time.time()
                #pnet dont use landmark
#                print("image: ", im)
                boxes, boxes_c,landmark = self.detect_pnet(im)
                t1 = time.time() - t
                sum_time += t1
                if boxes_c is None:
                    print('boxes_c is None...')
                    all_boxes.append(np.array([]))
                    #IMPORTANT IMPORTNAT IMPORTANT, still initialize landmark
                    landmarks.append(np.array([]))
                    batch_idx += 1
                    continue
                print("pnet_ouput boxes: ", boxes.shape)
                print("pnet_ouput boxes_c: ", boxes_c.shape)
            #rnet
            t2 =0
            if self.rnet_detector:
                t = time.time()
                boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
                t2 = time.time() -t
                sum_time += t2
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    batch_idx +=1
                    continue
                print("rnet_ouput boxes: ", boxes.shape)
                print("rnet_ouput boxes_c: ", boxes_c.shape)
            #onet
            t3 = 0
            if self.onet_detector:
                t = time.time()
                boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
                t3 = time.time() -t
                sum_time += t3
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    batch_idx+=1
                    continue
            print("onet_ouput boxes: ", boxes.shape)
            print("onet_ouput boxes_c: ", boxes_c.shape)
            print("onet_ouput landmark: ", landmark.shape)
            print("time cost " + '{:.3f}'.format(sum_time) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,t3))
            all_boxes.append(boxes_c)
            landmarks.append(landmark)
            batch_idx +=1

        return all_boxes, landmarks

    def get_face_from_single_image(self,image):
        images = np.array(image)
        boxes_c,landmarks = self.detect(images)
        rets = []
        for i in range(boxes_c.shape[0]):
            bbox=boxes_c[i,:4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            print(corpbbox)
            rets.append(image[corpbbox[0]:corpbbox[2], corpbbox[1]:corpbbox[3]].copy())
        return rets