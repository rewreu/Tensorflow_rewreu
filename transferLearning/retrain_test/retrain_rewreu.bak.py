import tensorflow as tf
from IOutil import create_image_lists
from random import shuffle

epochs = 10


def loadBaseModel2Graph(model_file=u"../inception_v3_model/classify_image_graph_def.pb", load_binary=False):
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


def ModifiedModel(graph=tf.get_default_graph()):
    with graph.as_default():
        input_layer = graph.get_tensor_by_name("DecodeJpeg/contents:0")
        connectLayer = graph.get_tensor_by_name("pool_3/_reshape:0")
        weightO = tf.Variable(tf.truncated_normal(shape=[2048, 2], stddev=0.01))
        biasO = tf.Variable(tf.zeros([2]))
        predict = tf.nn.relu(tf.matmul(connectLayer, weightO) + biasO)
        label = tf.placeholder(tf.float32, [2, 1])
        # loss = tf.reduce_mean(predict, label)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=predict, labels=label)
        loss = tf.reduce_mean(cross_entropy)
        # loss = tf.reduce_mean(tf.abs(predict - label))
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=[weightO, biasO])
        # optimizer.minimize(loss, var_list=[weightO, biasO])
        return graph, input_layer, label, optimizer, loss, predict


#
# def getBottleNeckVec(graph):
#     bottleneck_tensor = graph.get_tensor_by_name('pool_3/_reshape:0')


def main():
    graph2 = loadBaseModel2Graph()
    (graph, input_layer, label, optimizer, loss, predict) = ModifiedModel(graph2)
    base_dir = u"../food_dir"
    imagelist = create_image_lists(base_dir, testing_percentage=10, validation_percentage=20)
    encoding = {k: v for v, k in enumerate(imagelist.keys())}  # {"edamame": 0, "omelette": 1}
    category_size = len(encoding)
    training_list = []
    for category in imagelist:
        code = encoding[category]
        for image in imagelist[category]["training"]:
            path = base_dir + "/" + category + "/" + image
            training_list.append([path, code])
    shuffle(training_list)

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()

        for step in range(epochs):
            for i, item in enumerate(training_list):
                with open(item[0], "r") as f:
                    image = f.read()
                lb = sess.run(tf.one_hot(item[1], category_size))
                lb = lb.reshape(category_size, -1)
                feed_dict = {input_layer: image, label: lb}
                _, l, predictions = sess.run([optimizer, loss, predict], feed_dict=feed_dict)
                if i % 10 == 0:
                    print("loss at step %d is %f" % (step * len(training_list) + i, l))


if __name__ == "__main__":
    main()
