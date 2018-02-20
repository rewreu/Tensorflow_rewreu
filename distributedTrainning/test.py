import tensorflow as tf
FLAGS= tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("index", "", "Either 'ps' or 'worker'")
print FLAGS.index
FLAGS.index="uk"
print FLAGS.index