# using the inceptionV3 model from https://github.com/tensorflow/models/tree/master/research/slim
# to run this, put the code in tensorflow/models/research/slim/ folder
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
import numpy as np
from tensorflow import graph_util

height = 299
width = 299
channels = 3

# Create graph
X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points = inception_v3(X, num_classes=1001,is_training=False)
predictions = end_points["Predictions"]
saver = tf.train.Saver()

X_test = np.ones((1,299,299,3))  # a fake image, you can use your own image

# Execute graph
with tf.Session() as sess:
    saver.restore(sess, "./inception-v3_model_file/inception_v3.ckpt")
    predictions_val = predictions.eval(feed_dict={X: X_test})
    tf.train.write_graph(sess.graph_def, './', 'inceptionv3.pbtxt')
    graph=tf.get_default_graph()

    #output_node_names example,\
    output_node_names = "InceptionV3/Predictions/Reshape_1"
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        graph.as_graph_def(),
        output_node_names.split(",")  # The output node names are used to select the usefull nodes
    )
    model = output_graph_def.SerializeToString()
    with open("./inception_v3_batch.pb","wb") as modelfile:
        modelfile.write(model)