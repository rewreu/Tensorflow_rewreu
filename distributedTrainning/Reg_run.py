gpudb_host = 'localhost'
gpudb_port = 9191

## UDF Properties
proc_name = 'distributed_mnist_train'
file_names = ['MNIST.py']
command = 'python'
# args=[]
args = ['MNIST.py']
execution_mode = 'distributed'
options = {}

import collections
import time
import gpudb
from datetime import datetime, timedelta

files = {}
for ifile in file_names:
    with open(ifile, 'rb') as f:
        files[ifile] = f.read()

## Init connection to GPUdb
h_db = gpudb.GPUdb(encoding='BINARY', host=gpudb_host, port=gpudb_port)

## Drop older version of proc with same name if it already exists
if h_db.has_proc(proc_name)['proc_exists']:
    response = h_db.delete_proc(proc_name)
    if response['status_info']['status'] == 'OK':
        print('Dropping older version of proc')
    else:
        print('Error dropping older verison of proc')
        print(response)

## Register the UDF
response = h_db.create_proc(proc_name, execution_mode, files, command, args, options)

## Check the registration status
if response['status_info']['status'] == 'OK':
    print('Proc was created')
else:
    print('Error creating proc')
    print(response)

response = h_db.execute_proc(proc_name)

## Track the proc while it is running
start_time = datetime.now()
if response['status_info']['status'] == 'OK':
    run_id = response['run_id']
    print('Proc was launched successfully with run_id: ' + run_id)
    while h_db.show_proc_status(run_id)['overall_statuses'][run_id] == 'running':
        time.sleep(1)
        # print 'process is running... '
    final_proc_state = h_db.show_proc_status(run_id)['overall_statuses'][run_id]
    print("total running time is ", datetime.now() - start_time)
    print('Final Proc state: ' + final_proc_state)
    if final_proc_state == 'error':
        raise RuntimeError('proc error')
else:
    print('Error launching proc; response: ')
    print(response)
    raise RuntimeError('proc error')