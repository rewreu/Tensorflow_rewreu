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


def ModifiedModel(graph, labelsize):
    with graph.as_default():
        input_layer = graph.get_tensor_by_name("DecodeJpeg/contents:0")
        connectLayer = graph.get_tensor_by_name("pool_3/_reshape:0")
        weightO = tf.Variable(tf.truncated_normal(shape=(2048, labelsize), stddev=0.01), name="weightOL")
        biasO = tf.Variable(tf.zeros([labelsize, 1]), name="biasO")

        predict = tf.matmul(connectLayer, weightO) + biasO

        label = tf.placeholder(tf.int32, shape=(labelsize, 1))

        cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=predict, labels=label)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=[weightO, biasO])
        return graph, input_layer, label, optimizer, loss, predict

def main():
    base_dir = u"../food_dir"
    # base_dir = u"/notebooks/TransferLearning/fooddata/food-101/images"
    imagelist = create_image_lists(base_dir, testing_percentage=10, validation_percentage=20)
    encoding = {k: v for v, k in enumerate(imagelist.keys())}  # {"edamame": 0, "omelette": 1}
    category_size = len(encoding)
    training_list = []
    for category in imagelist:
        code = encoding[category]
        for image in imagelist[category]["training"]:
            path = base_dir + "/" + category.replace(" ", "_") + "/" + image
            training_list.append([path, code])
    shuffle(training_list)

    graph2 = loadBaseModel2Graph()
    (graph, input_layer, label, optimizer, loss, predict) = ModifiedModel(graph2, category_size)

    biasO = graph.get_tensor_by_name("biasO:0")
    weightO = graph.get_tensor_by_name("weightOL:0")
    connectLayer = graph.get_tensor_by_name("pool_3/_reshape:0")
    # print "train list is ",training_list
    with tf.Session(graph=graph) as sess:
        # tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        for step in range(epochs):
            for i, item in enumerate(training_list):
                with open(item[0], "r") as f:
                    image = f.read()
                lb = sess.run(tf.one_hot(item[1], category_size))
                lb = lb.reshape(category_size, -1)
                feed_dict = {input_layer: image, label: lb}
                _, l, predictions, b0, w0 = sess.run([optimizer, loss, predict, biasO, weightO], feed_dict=feed_dict)
                if i % 20 == 0:
                    print("loss at step %d is %f" % (step * len(training_list) + i, l))
                    #print w0[990:1000] monitor the weights change during optimizing

if __name__ == "__main__":
    main()
