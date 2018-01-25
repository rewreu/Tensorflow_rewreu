# from retrain import create_image_lists
import tensorflow as tf
import os
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile

if __name__ == "__main__":
    # ret=create_image_lists("./testImages",50,30)
    INCEPTION_LOG_DIR = '/tmp/tensorflow'

    if not os.path.exists(INCEPTION_LOG_DIR):
        os.makedirs(INCEPTION_LOG_DIR)
    with tf.Session() as sess:
        model_filename = './inception_v3_model/classify_image_graph_def.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        writer = tf.summary.FileWriter(INCEPTION_LOG_DIR, tf.get_default_graph())
        writer.close()

