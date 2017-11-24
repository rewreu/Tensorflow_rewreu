# Create event from pb file so that tensorboard can visualise,
# the pb file is from protobuf, one tested example is 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# to run the tensorboard, use:
# tensorboard --logdir=run1:/tmp/tensorflow/ --port 9292
#
import os
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile

INCEPTION_LOG_DIR ='/tmp/tensorflow'

if not os.path.exists(INCEPTION_LOG_DIR):
    os.makedirs(INCEPTION_LOG_DIR)

if __name__=="__main__":
    with tf.Session() as sess:
        model_filename = './model/saved_model.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        writer = tf.summary.FileWriter(INCEPTION_LOG_DIR, graph_def)
        writer.close()