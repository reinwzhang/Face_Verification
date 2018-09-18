# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:26:13 2018

@author: rein9
"""
import tensorflow as tf
import numpy as np
import os
import cv2
class Detector(object):
    '''
    net factory: rnet or onet
    datasize: 24 or 48
    figureout the landmark
    '''
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name = 'input_image')
            #cls:2, bbox:4, landmark:10
            self.cls_prob,self.bbox_pred, self.landmark_pred=net_factory(self.image_op, training=False)
            self.sess=tf.Session(config = tf.ConfigProto(allow_soft_placement=True,gpu_options=tf.GPUOptions(allow_growth = True)))
            saver=tf.train.Saver()
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print('Model Path: %s' % model_path)
            readstate=ckpt and ckpt.model_checkpoint_path
            assert readstate, 'the parameter dictionary is not valid'
            print('restore model parameter')
            saver.restore(self.sess, model_path)

        self.data_size= data_size
        self.batch_size= batch_size

    def get_model_filenames(self, model_dir):
        ''' Returns the path of the meta file and the path of the checkpoint file.
        Parameters:
        model_dir: (string),  the path to model dir.

        Returns:
        meta_file: (string), the path of the meta file
        ckpt_file: (string), the path of the checkpoint file
        '''
        #bookmark 09/02
        files = os.lisdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) ==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) >1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt in ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

    def predict(self, databatch):
        '''
        access data
        databatch: N*3*data_size*data_size(remember we reshaped channel to the front)
        '''
        batch_size = self.batch_size
        n = databatch.shape[0] # total num of data
        cur = 0
        minibatch = []
        while cur < n:
            minibatch.append(databatch[cur:min(cur+batch_size,n), :, :, :])
            cur += batch_size
        cls_prob_list=[]
        bbox_pred_list=[]
        landmark_pred_list=[]

        for idx, data in enumerate(minibatch):
            m= data.shape[0]
            real_size=self.batch_size
            if m < batch_size:
                #the last batch
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >=len(keep_inds):
                  # recursively shrink so the gap is smaller than keep_inds
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap !=0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data=data[keep_inds]
                real_size = m

                #cls_prob: num_batch*batch_size*2, bbox_pred: num_batch*batch_size*4, landmark_pred:num_batch*batch_size*10
                cls_prob, bbox_pred, landmark_pred= self.sess.run([self.cls_prob,self.bbox_pred,self.landmark_pred], feed_dict={self.image_op:data})
                cls_prob_list.append(cls_prob[:real_size])
                bbox_pred_list.append(bbox_pred[:real_size])
                landmark_pred_list.append(landmark_pred[:real_size])

        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)

    def predict_embedding(self, image):
        """Get the embedding vector of face by facenet
        Parameters:
        image: numpy array; input image array

        Returns:
        embedding: numpy array; the embedding vector of face
        """

        # !!!!!!!!!!!!!!!!!! Implement here !!!!!!!!!!!!!!!
        image_placeholder = self.sess.graph.get_tensor_by_name('input:0')
        embedding = self.sess.graph.get_tensor_by_name('fc_flatten:0')
        phase_train_placeholder = self.sess.graph.get_tensor_by_name('phase_train:0')
        image = cv2.resize(image, (160,160))
        feed_dict = {image_placeholder: np.stack([image]), phase_train_placeholder:False}
        emb = self.sess.run(embedding, feed_dict=feed_dict)
        return emb[0, :]
