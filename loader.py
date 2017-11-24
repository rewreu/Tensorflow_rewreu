import tensorflow as tf
import numpy as np
tmp=np.random.rand(1280,5)# get new array as input

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["haha"], './BuilderSavedModel/')
    graph = tf.get_default_graph()
    tf_train_dataset = graph.get_tensor_by_name("input:0")

    train_prediction=graph.get_tensor_by_name("output:0")
    w1=graph.get_tensor_by_name("w1:0")

    op=sess.run([train_prediction],feed_dict={tf_train_dataset : tmp})
    print op
    print sess.run(w1)

import os

import sys
sys.path.append('/usr/local/cuda-8.0/bin')
sys.path.append('usr/local/cuda-8.0/lib64')
usr/local/cuda-8.0/lib64