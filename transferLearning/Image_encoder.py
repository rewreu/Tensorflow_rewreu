import tensorflow as tf
from tensorflow import gfile
import numpy as np
input_depth=3
input_height=299
input_width=299
input_mean = 127.5
input_std = 127.5
tf.reset_default_graph()
graph = tf.Graph()

with tf.Session() as sess:
    # model_filename = FLAGS.model_file

    with gfile.FastGFile("./inception_v3_model/classify_image_graph_def.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
graph2=tf.get_default_graph()

with graph2.as_default():
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)

init = tf.global_variables_initializer()

with tf.Session(graph=graph2) as sess:
    # sess.run(init)
    f=open("./testImages/cat/cat1.jpeg").read()
    out=sess.run([mul_image],feed_dict={jpeg_data:f})
    # out=np.squeeze(out,0)
    print out
    bottleneck_tensor = graph2.get_tensor_by_name('pool_3/_reshape:0')
    resized_input_tensor = graph2.get_tensor_by_name('Mul:0')
    image2048 = sess.run([bottleneck_tensor], feed_dict={resized_input_tensor: out[0]})
    print image2048

with tf.Session(graph=graph2) as sess:
    # sess.run(init)
    f=open("./testImages/cat/cat1.jpeg").read()
    f2 = open("./testImages/cat/cat1.jpeg").read()
    imageInput = graph2.get_tensor_by_name('DecodeJpeg/contents:0')
    image2048 = sess.run([bottleneck_tensor], feed_dict={imageInput: f})
    print image2048


#
# with tf.Session(graph = graph2) as sess:
#     bottleneck_tensor = graph2.get_tensor_by_name('pool_3/_reshape:0')
#     resized_input_tensor = graph2.get_tensor_by_name('Mul:0')
#     image2048=sess.run([bottleneck_tensor],feed_dict={resized_input_tensor:out[0]})
#     print image2048