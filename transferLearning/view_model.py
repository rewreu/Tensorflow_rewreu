# to use this,
# run python view_model.py --model_file model.pb --log_dir /tmp
# then go to the default port for tensorboard to view your model(normally 6006)
import argparse
import os
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile

if __name__ == "__main__":
    FLAGS=None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_file',
        type=str,
        default='./inception_v3_model/classify_image_graph_def.pb',
        help='Path to model file, in pb format.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow',
        help='Path to folder to log for tensorboard.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    LOG_DIR = FLAGS.log_dir
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with tf.Session() as sess:
        model_filename = FLAGS.model_file

        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
        writer.close()
    os.system('tensorboard --logdir=' + LOG_DIR)