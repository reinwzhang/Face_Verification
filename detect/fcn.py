# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:43:06 2018

@author: rein9
"""
import tensorflow as tf
from .MTCNN_config import config

class FcnDetector(object):
    def __init__(self, net_factory, model_path):
        # initialize tf graph
        graph = tf.Graph()
        with graph.as_default():
            self.img_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')

            image_reshape = tf.reshape(self.img_op, [1, self.height_op, self.width_op, 3])
            # construct a model here to get the cls_map and bbox
            self.cls_prob, self.bbox_pred,_ = net_factory(image_reshape, training=False)
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            # save the graph
            saver = tf.train.Saver()
            #read in the model, which contains the information about the ckpt state
            model_dict = '/'.join(model_path.split('/')[:-1])
            print("FCN model dict: ", model_dict)
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print("FCN model path: ", model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)

    def predict(self, datapath):
        height,width,_ = datapath.shape
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                               feed_dict={self.img_op:datapath,
                                                          self.width_op:width,
                                                          self.height_op:height})
        return cls_prob, bbox_pred