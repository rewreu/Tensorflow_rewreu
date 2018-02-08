import tensorflow as tf
from IOutil import encodeRecursive
"""
Modify these according to your local setup
BASE_MODEL is the protobuf file which has the image preprocessing pipeline
IMAGE_DIR is the folder has images, has the following structure:
    images/
    |-- apple_pie
    │   |-- 1005649.jpg
    │   |-- 101251.jpg
    |-- hot_dog
    │   |-- 1011328.jpg
EXPORT_DIR is where the encoded images exports. It has the same structure as IMAGE_DIR, 
with suffix as txt instead of jpg
"""
BASE_MODEL = u"./inception_v3_model/classify_image_graph_def.pb"
IMAGE_DIR = "./fooddata/food-101/images"
EXPORT_DIR = "./fooddata/food-101-encode"


def loadModel2graph(model_file=BASE_MODEL, load_binary=False):
    if load_binary:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file)
    else:
        with tf.gfile.Open(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    graph = tf.get_default_graph()
    return graph


def run_encodeRecursive():
    tf.reset_default_graph()
    graph = loadModel2graph()
    with tf.Session(graph=graph) as sess:
        input = graph.get_tensor_by_name("DecodeJpeg/contents:0")
        output = graph.get_tensor_by_name("pool_3/_reshape:0")
        encodeRecursive(IMAGE_DIR, EXPORT_DIR, sess, input, output)


if __name__ == "__main__":
    run_encodeRecursive()
