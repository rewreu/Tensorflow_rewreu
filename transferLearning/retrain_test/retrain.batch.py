"""
Using transfer learning to retrain image classification model

"""
import tensorflow as tf
from IOutil import create_image_lists, saveModel
from random import shuffle
import numpy as np

EPOCHS = 10
BATCH_SIZE = 100
BASE_MODEL_PATH = u"../inception_v3_model/classify_image_graph_def.pb"
IMAGE_DIR = u"../food_dir_encoded"


def loadBaseModel2Graph(model_file=BASE_MODEL_PATH, load_binary=False):
    if load_binary:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file)
    else:
        with tf.gfile.Open(model_file, 'r') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name="")
    graph = tf.get_default_graph()
    return graph


def ModifiedModel(graph, labelsize):
    """

    :param graph: tenforflow graph loaded from base model
    :param labelsize: number of catergories to classify
    :return: graph: modfied graph, input_layer: the endpoint where imagesVec feed into,
            label: endpoint where label feed into, optimizer, loss,
            predict: the predict result for training purpose(output vector is for inferencing)

    """
    with graph.as_default():
        connectLayer = graph.get_tensor_by_name("pool_3/_reshape:0")
        input_layer = tf.placeholder(tf.float32, shape=[None, 2048])  # can be multiple at same time

        weightO = tf.Variable(tf.truncated_normal(shape=(2048, labelsize), stddev=0.01), name="weightO")
        biasO = tf.Variable(tf.random_normal([labelsize]), name="biasO")

        predict = tf.add(tf.matmul(input_layer, weightO) , biasO)

        output = tf.add(tf.matmul(input_layer, weightO) , biasO, name="outputO")
        label = tf.placeholder(tf.int32, shape=(None, labelsize))

        #cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=predict, labels=label)
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=label)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=[weightO, biasO])
        return graph, input_layer, label, optimizer, loss, predict


def feedVec(offset, batch_size, shuffled_image_path_list):
    """

    :param offset:
    :param batch_size:
    :param shuffled_image_path_list: a list comes from shuffleList function
    :return: list_encoded_images: a list of bottleneck_values(list of float numbers),
             labels: a list of labels
             both are ready to feed into tensorflow
    """

    list_encoded_images = []
    labels = []
    for item in shuffled_image_path_list[offset:offset + batch_size]:
        with open(item[0], 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        list_encoded_images.append(bottleneck_values)
        labels.append(item[1])

    return np.asarray(list_encoded_images, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def shuffleList(imagelist, base_dir, imageSet="training"):
    """

    :param imagelist: image list returns from create_image_lists call,
    :param base_dir: base directory contains all image folders
    :param imageSet: options available are "training","testing", "validation"
    :return: shuffled_image_path_list: a shuffled list, contains [[path_to_encoded_image, code],[]]
             encoding_dict: reversed decoding dictionary, {0: "edamame", 1: "omelette"}
    """
    encoding_dict = {k: v for v, k in enumerate(imagelist.keys())}  # {"edamame": 0, "omelette": 1}
    # category_size = len(encoding_dict)
    shuffled_image_path_list = []
    for category in imagelist:
        code = encoding_dict[category]
        for image in imagelist[category][imageSet]:
            path = base_dir + "/" + imagelist[category]["dir"] + "/" + image
            shuffled_image_path_list.append([path, code])

    # reverse keys values pairs{0: "edamame", 1: "omelette"}
    decoding_dict = dict((v, k) for k, v in encoding_dict.iteritems())

    shuffle(shuffled_image_path_list)
    return shuffled_image_path_list, decoding_dict


def main():
    base_dir = IMAGE_DIR
    # base_dir = u"/notebooks/TransferLearning/fooddata/food-101/images"
    imagelist = create_image_lists(base_dir, testing_percentage=5, validation_percentage=5, load_raw_image=False)
    training_suffledlist, decoding_dict = shuffleList(imagelist, base_dir, imageSet="training")
    testing_suffledlist, _ = shuffleList(imagelist, base_dir, imageSet="testing")
    validation_suffledlist, _ = shuffleList(imagelist, base_dir, imageSet="validation")
    category_size = len(decoding_dict.keys())
    graph2 = loadBaseModel2Graph()

    (graph, input_layer, label_layer, optimizer, loss, predict) = ModifiedModel(graph2, category_size)

    biasO = graph.get_tensor_by_name("biasO:0")
    weightO = graph.get_tensor_by_name("weightO:0")
    # print "train list is ",training_list

    train_step = int(len(training_suffledlist) / BATCH_SIZE)
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for epoch in range(EPOCHS):
            for step in range(train_step):
                data, label = feedVec(step * BATCH_SIZE, BATCH_SIZE, training_suffledlist)
                label=sess.run(tf.one_hot(label, category_size))
                feed_dict = {input_layer: data, label_layer: label}
                _, l, predictions, b0, w0 = sess.run([optimizer, loss, predict, biasO, weightO], feed_dict=feed_dict)
                if step % 10 == 0:
                    print("Training loss at EPOCH %d, step %d is %f" % (epoch, step * BATCH_SIZE, l))

            # show validation loss
            data, label = feedVec(0, 10 * 4, validation_suffledlist[:100]) # limit the validation len to 100 in case
                                                                            # of out of GPU memory
            label = sess.run(tf.one_hot(label, category_size))
            feed_dict = {input_layer: data, label_layer: label}
            l, b0, w0 = sess.run([loss, biasO, weightO],
                                 feed_dict=feed_dict)
            print("Validation loss at EPOCH %d, step %d is %f" % (epoch, step * BATCH_SIZE, l))
            print("Weights patial is:\n,", w0[990:1000])
            print("Bias patial is:\n,", b0[990:1000])

            # print w0[990:1000] monitor the weights change during optimizing
        # show test loss
        data, label = feedVec(0, 10 * 4, testing_suffledlist[:100])
        label = sess.run(tf.one_hot(label, category_size))
        feed_dict = {input_layer: data, label_layer: label}
        l, b0, w0 = sess.run([loss, biasO, weightO],
                             feed_dict=feed_dict)
        print("test loss is" % (l))
    # save model to load probuf file
    output_node_names = "DecodeJpeg/contents:0,outputO"
    saveModel(sess=sess, graph=graph, output_node_names=output_node_names)


if __name__ == "__main__":
    main()
