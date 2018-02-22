#%%writefile MNIST.py
### Between graph replication
### Asynchronus training
'''
Distributed Tensorflow 1.2.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server.
Change the hardcoded host urls below with your own hosts.
Run like this:
pc-01$ python MNIST.py --job_name="ps" --task_index=0
pc-02$ python MNIST.py --job_name="worker" --task_index=0
pc-03$ python MNIST.py --job_name="worker" --task_index=1
'''

from __future__ import print_function

# import tensorflow as tf
import sys
import time
from kinetica_proc import ProcData
proc_data = ProcData()
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# cluster specification
parameter_servers = ["localhost:1988"]
workers = ["localhost:1989","localhost:1990","localhost:1991"]
#cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# input flags
# tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
# tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
# FLAGS = tf.app.flags.FLAGS
print("------------0")

# specify according to rank # and tom #
#
# Toms_per_rank=2
# if proc_data.request_info["rank_number"]==str(0):
#     pass
# if proc_data.request_info["rank_number"]=="1" and proc_data.request_info["tom_number"]=="0":
#     FLAGS.job_name='ps'
#     FLAGS.task_index=0
# if proc_data.request_info["rank_number"]=="1" and proc_data.request_info["tom_number"]=="1":
#     FLAGS.job_name='worker'
#     FLAGS.task_index=0
# if proc_data.request_info["rank_number"]=="2" and proc_data.request_info["tom_number"]=="0":
#     FLAGS.job_name='worker'
#     FLAGS.task_index=1
# if proc_data.request_info["rank_number"]=="2" and proc_data.request_info["tom_number"]=="1":
#     FLAGS.job_name='worker'
#     FLAGS.task_index=2

print("---------------------")
print("rank %s, tom %s" %(proc_data.request_info["rank_number"],proc_data.request_info["tom_number"]))
#print("job_name %s, task_index %s" % (FLAGS.job_name,FLAGS.task_index))
# start a server for a specific task
# server = tf.train.Server(
#     cluster,
#     job_name=FLAGS.job_name,
#     task_index=FLAGS.task_index)
print("------------1.5")
# config
batch_size = 100
learning_rate = 0.0005
training_epochs = 20

proc_data.complete()