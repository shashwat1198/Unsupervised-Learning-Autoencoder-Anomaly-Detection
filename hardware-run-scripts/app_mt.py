'''
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import sys
if '/usr/lib/python3.6/site-packages' not in sys.path:
        sys.path.append('/usr/lib/python3.6/site-packages')

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
#import sys
import argparse
import subprocess
#import pynq


divider = '------------------------------------'

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id,start,dpu,img):

    '''get tensor'''
    yz = np.zeros([2000,1200]) 	
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    #print('***Input tensor info***')
    #print(inputTensors)
    #print(input_ndim)
    #print('***Output tensor info***')
    #print(outputTensors)
    #print(output_ndim)
    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    t3 = 0
    write_index = start
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count
        '''prepare batch input/output '''
        outputData = []
        inputData = []
       # t1 = time.time()
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]#The datatype has to be float32, int8n does not work.
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]
        #print(len(outputData))
        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        #print(inputData)
        t1 = time.time()
        job_id = dpu.execute_async(inputData,outputData)
        #t1 = time.time()
        dpu.wait(job_id)
        t2 = time.time()
        print(t2-t1)
        t3 = t3 + t2-t1
        #print('-----=',t3)
        '''store output vectors '''
        #print(runSize)
        for j in range(runSize):
            out_q[write_index] = outputData
            #print(outputData)
            #yyz = np.array(out_q[write_index])
            #yz[write_index] = yyz.flatten()
            #for k in range(len(yz)):
            #    print(str(yz[k])+',')
            #print("\n")
            write_index += 1
        count = count + runSize
    print("------ Real ----",t3)
    #np.savetxt("gear_output.csv", yz, fmt="%d", delimiter=",") #This command write the output from the numpy array to a text file in interger format with ',' as the delimiter.


def app(image_dir,threads,model):

    #listimage=os.listdir(image_dir)
    #runTotal = len(listimage)
    runTotal = 2000

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))


    ''' preprocess images '''
    print (divider)
    print('Pre-processing',runTotal,'images...')
    img = []
   #filenames_X = ["autoencoder_id_train.txt"]#gear_autoencoder_test
   # filenames_X = ["dos_big_id.txt"]#gear_autoencoder_test
    filenames_X = ["gear_big_id.txt"]#gear_autoencoder_test
   # X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])
    X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])
    filenames_Y = ["Fuzzy_Y.txt"]
    Y_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Y])
    a = np.array(X_set)
    print(a.shape)
    b = np.array(Y_set)
    print(b.shape)

    #Train labels
    samples= 200000
    dif = 0 #800000 #900000
    # Hyperparameters
    nset = 100#76
    n_steps = nset
    n_features = 12#76 #76
    n_batches = int(samples/nset)

    test_X = np.zeros([n_batches,nset,n_features])#10
    cc = np.zeros([n_batches,1])
    c = np.zeros([n_batches,1])
    test_Y = np.zeros([n_batches,1])
    for i in range(n_batches):
        for j in range(nset):
            for k in range(n_features):#11
                test_X[i][j][k] = a[i*nset+j+dif][k]

    test_X = test_X.reshape([n_batches,1,100,12,1])#11
    #test_Y = test_Y.reshape([n_batches,2])
    for i in range(runTotal):
    #    path = os.path.join(image_dir,listimage[i])
        img.append(test_X[i])

    '''run threads '''
    global a_file 
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end
    print('*** Time start ***')
    #t1 = np.zeros(50000)
    #t2 = np.zeros(50000)
    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print (divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))
    print (divider)

    return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')  
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='customcnn.xmodel', help='Path of xmodel. Default is customcnn.xmodel')

  args = ap.parse_args()

  print(divider)  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)

  app(args.image_dir,args.threads,args.model)

if __name__ == '__main__':
  main()

