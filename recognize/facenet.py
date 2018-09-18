import tensorflow as tf
import numpy as np
import os
import re
import cv2

class FaceNet(object):
    def __init__(self, model_path):
        # Read model files and init the tf graph and model
        # !!!!!!!!!!!!!!!!!! Implement here !!!!!!!!!!!!!!!
        ### init
        graph = tf.Graph()
        with graph.as_default():
          self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True,gpu_options=tf.GPUOptions(allow_growth=True)))
          print("Model Path: %s" % model_path)
          # read in the meta file
          meta_file, ckpt_file = self.get_model_filenames(model_path)
          # a new way of importing meta graph
          saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file), input_map=None)
          saver.restore(self.sess, os.path.join(model_path, ckpt_file))

    def get_model_filenames(self, model_dir):
        """ Returns the path of the meta file and the path of the checkpoint file.

        Parameters:
        ----------
        model_dir: string
            the path to model dir.

        Returns:
        -------
        meta_file: string
            the path of the meta file
        ckpt_file: string
            the path of the checkpoint file
        """
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file


    def predict(self, image):
        """Get the embedding vector of face by facenet
        Parameters:
        image: numpy array; input image array

        Returns:
        embedding: numpy array; the embedding vector of face
        """

        # !!!!!!!!!!!!!!!!!! Implement here !!!!!!!!!!!!!!!
        image_placeholder = self.sess.graph.get_tensor_by_name('input:0')
        embedding = self.sess.graph.get_tensor_by_name('embeddings:0')
        phase_train_placeholder = self.sess.graph.get_tensor_by_name('phase_train:0')
        image = cv2.resize(image, (160,160))
        feed_dict = {image_placeholder: np.stack([image]), phase_train_placeholder:False}
        emb = self.sess.run(embedding, feed_dict=feed_dict)
        return emb[0, :]