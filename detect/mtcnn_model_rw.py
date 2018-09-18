# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 12:52:34 2018

@author: rein9
"""

import tensorflow as tf
from tensorflow.contrib import slim
#import numpy as np
num_keep_radio = 0.7
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def cal_accuracy(cls_prob, label):
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int32)
    cond = tf.where(tf.greater_qeual(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked),tf.float32))
    return accuracy_op


def cls_ohem(cls_prob, label):
    '''
    returns: -(yilog(pi) + (1-yi)log(1-pi))
    '''
    zeros = tf.zeros_like(label)
    # when label < 0, invalid so convert them to zeros
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob=tf.size(cls_prob)
    cls_prob_reshape=tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int=tf.cast(label_filter_invalid, tf.int32)
    num_row=tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)

    zeros = tf.zeros_like(label_prob, dtype=tf.ffloat32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    valid_inds = tf.where(label<zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio, dtype=tf.int32)

    loss = loss*valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)

def bbox_ohem(bbox_pred, bbox_target, label):
    '''
    bbox loss it to calculate the ohem distance between bbox_pred and GT
    '''
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    square_error = tf.reduce_sum(tf.sqaure((bbox_pred -bbox_target), axis=1))
    #keep num scalar
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid,dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.readuce_mean(square_error)

def landmark_ohem(landmark_pred, landmark_target, label):
    #assumption keep label = -2 then do landmark detection
    ones = tf.ones_like(label, dtype =tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    square_error = tf.reduce_sum(tf.square(landmark_pred, landmark_target), axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _,k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_sum(square_error)

def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    #why activation is prelu, why?
    '''
    leaky relu vs prelu:
      https://datascience.stackexchange.com/questions/18583/what-is-the-difference-between-leakyrelu-and-prelu
      Leaky ReLUs: allow a small, non-zero gradient when the unit is not active.
      Parametric ReLUs: take this idea further by making the coefficient of leakage into a parameter
                        that is learned along with the other neural network parameters.
    '''
    with slim.arg_scope([slim.conv2d],
                         activation_fn=prelu,
                         weights_initializer=slim.xavier_initializer(),
                         biases_initializer=tf.zeros_initializer(),# slim does not have zeros initilizer
                         weights_regularizer=slim.l2_regularizer(0.0005),
                         padding='valid'):
        print("PNet input shape: ", inputs.get_shape())
        net=slim.conv2d(inputs,num_outputs=10,kernel_size=[3,3],stride=1,scope='conv1')
        print("PNet conv1 shape: ", net.get_shape())
        net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool1')
        print("PNet pool1 shape: ", net.get_shape())
        net=slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        print("PNet conv2 shape: ", net.get_shape())
        net=slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        print("PNet conv3 shape: ", net.get_shape())
        # final 3 conv to get H*W*2 classifier, H*W*4 bbox, H*W*10 landmar_pred
        conv4_1=slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        print('P_Net conv4_1 shape ',net.get_shape())
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)# important scope name should not be the same as veriable name
        print('P_Net bbox_pred conv layer shape ',bbox_pred.get_shape())
        landmark_pred=slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
        print('P_Net ladmark conv layer shape', landmark_pred.get_shape())

        if training:
            #batch*2 to determin if it is a face
            #why squeezing? what will happe
            cls_prob=tf.squeeze(conv4_1,[1,2],name='cls_prob')
            cls_loss=cls_ohem(cls_prob,label)
            #check bbox_loss
            bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
            #landmark loss
            landmark_pred=tf.squeeze(landmark_pred,[1,2],name='landmark_pred')
            landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
            accuracy=cal_accuracy(cls_prob,label)

            #tf.add_n: Adds all input tensors element-wise.
            L2_loss=tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            #test, batch_size=1
            cls_prob_test=tf.squeeze(conv4_1,axis=0)
            bbox_pred_test=tf.squeeze(bbox_pred,axis=0)
            landmark_pred_test=tf.squeeze(landmark_pred,axis=0)
            return cls_prob_test,bbox_pred_test,landmark_pred_test

def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        print("RNet input shape: ", inputs.get_shape())
        net = slim.conv2d(inputs,num_outputs=28,kernel_size=[3,3],stride=1,scope='conv1')
        print("RNet conv1 shape: ", net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool1',padding='SAME')
        print("RNet pool1 shape: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope='conv2')
        print("RNet conv2 shape: ", net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
        print("RNet pool2 shape: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope='conv3')
        print("RNet conv3 shape: ", net.get_shape())
        fc_flatten=slim.flatten(net)
        print("RNet fc1 shape: ", fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten,num_outputs=128,scope='fc1',activation_fn=tf.nn.relu)
        #binary classification
        print('RNet fc1 shape after flattening: ',fc1.get_shape())
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope='cls_fc',activation_fn=tf.nn.softmax)
        print('RNet cls_prob fc shape ',cls_prob.get_shape())
        #bounding box
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope='bbox_fc',activation_fn=None)
        print('RNet bbox_pred fc shape ',bbox_pred.get_shape())
        #landmark
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope='landmark_fc',activation_fn=None)
        print('RNet landmark fc shape ',landmark_pred.get_shape())

        if training:
            cls_loss=cls_ohem(cls_prob,label)
            bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
            accuracy=cal_accuracy(cls_prob,label)
            landmark_loss=landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss=tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred

def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        print("ONet input shape: ", inputs.get_shape())
        net = slim.conv2d(inputs,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv1')
        print("ONet conv1 shape: ", net.get_shape())
        # in the original model, for O net all pooling using stride of 2
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool1',padding='SAME')
        print("ONet pool1 shape: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope='conv2')
        print("ONet conv2 shape: ", net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
        print("ONet pool2 shape: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope='conv3')
        print("ONet conv3 shape: ", net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[2,2],stride=2,scope='pool3',padding='SAME')
        print("ONet pool3 shape: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope='conv4')
        print("ONet conv4 shape: ", net.get_shape())
        fc_flatten = slim.flatten(net)
        print("ONet fc input shape: ", fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten,num_outputs=256,scope='fc1',activation_fn=tf.nn.relu)
        #cls
        print('ONet fc shape after flattening: ',fc1.get_shape())
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope='cls_fc',activation_fn=tf.nn.softmax)
        print('ONet cls_prob fc shape ',cls_prob.get_shape())
        #bbox
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope='bbox_fc',activation_fn=None)
        print('ONet bbox_pred fc shape ',bbox_pred.get_shape())
        #landmark
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope='landmark_fc',activation_fn=None)
        print('ONet landmark fc shape ',landmark_pred.get_shape())
        if training:
            cls_loss=cls_ohem(cls_prob,label)
            bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
            accuracy=cal_accuracy(cls_prob,label)
            landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
        #landmark
