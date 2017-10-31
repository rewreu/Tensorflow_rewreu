import tensorflow as tf
import numpy as np
np.random.seed(42)
tf.set_random_seed(42)
mu, sigma, size = 0, 0.1, 10000
s1 = np.random.normal(mu, sigma, size)
s2 = np.random.normal(mu, sigma, size)
s3 = np.random.normal(mu, sigma, size)
s4 = np.random.normal(mu, sigma, size)
s5 = np.random.normal(mu, sigma, size)

s=0.25*s1+0.35*s2+0.15*s3+0.1*s4+0.15*s5

logi=1.0/(1+np.exp(-s))
logi[logi<0.5]=0
logi[logi>0.5]=1

logiErr=logi+np.random.normal(mu, sigma/5, size)
logiErr=(logiErr[:,None]).astype(np.float32)
c=np.column_stack((s1,s2,s3,s4,s5))
c=c.astype(np.float32)
print("shape of the label",logiErr.shape)
print("shape of the data",c.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

c,logiErr=randomize(c,logiErr)

# Split the train and validation
r =np.random.rand(logiErr.shape[0])
Y_train = logiErr[r<0.8]
X_train = c[r<0.8]
Y_valid = logiErr[r>=0.9]
X_valid = c[r>=0.9]
Y_test = logiErr[((0.8<=r)*(r<0.9))]
X_test = c[((0.8<=r)*(r<0.9))]

#!rm - rf. / BuilderSavedModel

num_labels = 1
num_hidden_nodes = 14
beta = 1e-3
batch_size = 1280

tf.reset_default_graph()

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, X_train.shape[1]), name="input")
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 1))
    tf_valid_dataset = tf.constant(X_valid)
    tf_valid_labels = tf.constant(Y_valid)
    tf_test_dataset = tf.constant(X_test)
    tf_test_labels = tf.constant(Y_test)
    weights_1 = tf.Variable(tf.truncated_normal([X_train.shape[1], num_hidden_nodes]), name="w1")
    weights_2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]), name="w2")
    biases_1 = tf.Variable(tf.zeros([14]), name="buss")
    biases_2 = tf.Variable(tf.zeros([1]), name="b2")
    b1 = tf.Variable(2.0, name="bias")
    # layer_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1),keep_prob=0.8)
    layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
    logits = tf.matmul(layer_1, weights_2) + biases_2
    loss = tf.reduce_mean(tf.abs(logits - tf_train_labels)) + beta * (
    tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2))

    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_prediction = tf.add(tf.matmul(layer_1, weights_2), biases_2, name="output")

    layer_1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
    valid_prediction = tf.matmul(layer_1_valid, weights_2) + biases_2
    valid_loss = tf.reduce_mean(tf.abs(valid_prediction - tf_valid_labels))

    layer_1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
    test_prediction = tf.add(tf.matmul(layer_1_test, weights_2), biases_2, name="output2")
    test_loss = tf.reduce_mean(tf.abs(test_prediction - tf_test_labels))

num_steps = 20

builder = tf.saved_model.builder.SavedModelBuilder('./BuilderSavedModel/')
with tf.Session(graph=graph) as session:
    # saver = tf.train.Saver()
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps + 1):
        offset = (step * batch_size) % (X_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :]
        batch_labels = Y_train[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        b_1 = session.run(biases_1)

    print(b_1)
    # with tf.Session() as session:
    # session.run(b11.initializer)
    builder.add_meta_graph_and_variables(session,
                                         ["haha"],
                                         signature_def_map=None,
                                         assets_collection=None)
builder.save()