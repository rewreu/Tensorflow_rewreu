import tensorflow as tf
from IOutil import create_image_lists
from IOutil import  encodeRecursive
def loadModel2graph(model_file=u"../inception_v3_model/classify_image_graph_def.pb", load_binary= False):
#    with tf.Session() as sess:
        if load_binary:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file)
        else:
            with tf.gfile.Open(model_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        #tf.import_graph_def(graph_def)
        _ = tf.import_graph_def(graph_def, name='')
        graph = tf.get_default_graph()
        return graph

def test_encodeRecursive():
    tf.reset_default_graph()
    graph=loadModel2graph()
    with tf.Session(graph= graph) as sess:
        input=graph.get_tensor_by_name("DecodeJpeg/contents:0")
        output=graph.get_tensor_by_name("pool_3/_reshape:0")
        encodeRecursive("../testImages", "../encodeImages", sess, input, output)
    print "done"

def test_load_image_list():
    imagelist=create_image_lists("../encodeImages", testing_percentage=20,
                                 validation_percentage=30, load_raw_image= False)
    # print len(imagelist["omelette"]["training"])
    # print len(imagelist["omelette"]["testing"])
    # print len(imagelist["omelette"]["validation"])
    print imagelist

if __name__ == "__main__":
    # test_encodeRecursive()


    test_load_image_list()
    print "done"
