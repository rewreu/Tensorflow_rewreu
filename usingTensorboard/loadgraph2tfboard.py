# example of using tensorboard to visualise the graph created.
# the graph below is just an random graph
import os
import tensorflow as tf
from tensorflow.python.platform import gfile

INCEPTION_LOG_DIR ='/tmp/tensorflow'

if not os.path.exists(INCEPTION_LOG_DIR):
    os.makedirs(INCEPTION_LOG_DIR)
if __name__=="__main__":
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
        with tf.name_scope("input_lower"):
            y = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
            y2 = tf.placeholder(tf.float32, shape=[784, 784], name="y2-input")
    with tf.name_scope("multi"):
        z=tf.nn.dropout(tf.matmul(x,y2),keep_prob=0.8)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(INCEPTION_LOG_DIR, tf.get_default_graph())
        writer.close()