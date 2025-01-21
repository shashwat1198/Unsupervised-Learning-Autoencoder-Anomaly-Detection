import numpy as np
import tensorflow as tf
import time
from scipy.spatial.distance import hamming

filenames_X = ["./autoencoder_id_train.txt"]
X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])
# filenames_X = ["./dos_big_id.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])
# filenames_X = ["./fuzzy_big_id.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])
# filenames_X = ["./rpm_big_id.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])
# filenames_X = ["./gear_big_id.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])
filenames_Y = ["./RPM_Y.txt"]
Y_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Y])
a = np.array(X_set)
print(a.shape)
b = np.array(Y_set)
print(b.shape)

from tensorflow_model_optimization.quantization.keras import vitis_quantize
with vitis_quantize.quantize_scope():
    model = tf.keras.models.load_model('./quantised_cae_v3_100.h5')
model.summary()

#Train labels
samples= 100000
dif = 850000 #850000 
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

test_X = test_X.reshape(n_batches,nset,n_features)
normal = 0
attack = 0
for i in range(n_batches):
    z = np.sum(b[i*nset:i*nset+nset])
    if(z < 1):#Normal
        test_Y[i][0] = 0
        normal = normal + 1
    if(z >= 1):#Attack
        test_Y[i][0]= 1
        attack = attack + 1

c = model.predict(test_X)
cc = c.round(decimals = 0)#Rounds values from 0.5 or less to 0 and 0.51 or more to 1
count = np.zeros(8) 
TN = 0
TP = 0
FN = 0
FP = 0
max_val = 0
threshold_const = 0
for j in range(20):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    max_val = 0
    for i in range(n_batches):
        y_true = test_X[i].flatten()
        y_pred = cc[i].flatten()
        # print(y_true)
        # print(y_pred)
        # mse = tf.keras.losses.MeanSquaredError()
        distance = hamming(y_true, y_pred) * len(y_true)
        # print(distance,test_Y[i])
        # print(distance)
        # print('----------')
        # a = mse(y_true, y_pred).numpy()
        if(distance > max_val):
            max_val = distance
        threshold = threshold_const+j
        # This is for attack datasets
        # if(distance <= threshold and test_Y[i] == 0):
        #     TN = TN + 1
        # elif(distance > threshold and test_Y[i] == 1):
        #     TP = TP + 1
        # elif(distance <= threshold and test_Y[i] == 1):
        #     FN = FN + 1
        # elif(distance > threshold and test_Y[i] == 0):
        #     FP = FP + 1
        # This is for normal message dataset
        if(distance <= threshold):
            TN = TN + 1
        elif(distance > threshold):
            FP = FP + 1

    # This is for the normal datasets
    # print('Threshold '+str(threshold))
    # print('TN = '+str(TN))
    # print('FP = '+str(FP))  
    print('Accuracy = '+str(TN))
    # print('Max Error = '+str(max_val))

    #This is for the attack datasets
    # print('Threshold '+str(threshold))
    # print('Normal messages = '+str(normal))
    # print('Attack messages = '+str(attack))
    # print('TN = '+str(TN))
    # print('TP = '+str(TP))
    # print('FN = '+str(FN))
    # print('FP = '+str(FP))
    # print('Accuracy = '+str(TN+TP))
    # print('Max Error = '+str(max_val))

    # print('----------------------------')