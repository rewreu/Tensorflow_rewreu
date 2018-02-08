import tensorflow as tf
from tensorflow import gfile
import sys
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(arg):
    with open("./model.dict", "r") as dictf:
        s = dictf.read()
    model_dict = json.loads(s)
    dict_len = len(model_dict.keys())
    topN = int(arg[1])
    if (len(arg) == 3 and topN > 0 and topN < dict_len / 2):
        pass
    else:
        topN = 5
    print("Input file path is: ", arg[2])

    model_dict = json.loads(s)
    with tf.Session() as sess:
        model_filename = "./model.pb"
        print("Loading model...")
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        print("Finished loading model...")
    graph = tf.get_default_graph()

    with tf.Session(graph=graph) as sess:
        inp = graph.get_tensor_by_name("DecodeJpeg/contents:0")
        outp = graph.get_tensor_by_name("outputO:0")
        with open(arg[2], "rb") as f:
            image = f.read()
        predict = sess.run([outp], feed_dict={inp: image})
        predict = sess.run(tf.nn.softmax(np.squeeze(predict)))
        top5 = predict.argsort()[-topN:][::-1]
        print("Prediction results: ")
        for i in top5:
            print("(" + model_dict[str(i)], ", probablity is " + str(predict[i])[:5], ")")


if __name__ == "__main__":
    main(sys.argv)

