import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import applications, Sequential, utils
from tensorflow.keras.layers import Conv1D, Conv2DTranspose, UpSampling2D, Reshape, Input, ReLU, Dense, BatchNormalization, Dropout , TimeDistributed, LSTM, Flatten, Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import L1L2

#Precision, recall, f1_score computation
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming

# filenames_X = ["normal_id_data_binary_train.csv"]
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["autoencoder_id_train.txt"]
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["autoencoder_gear_id_test.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["rpm_autoencoder_test.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["rpm_big_id.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["fuzzy_autoencoder_test.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["fuzzy_big_test.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["fuzzy_big_id.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["dos_autoencoder_test.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

filenames_X = ["dos_big_id.txt"]#gear_autoencoder_test
X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["gear_autoencoder_test.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["gear_big_id.txt"]#gear_autoencoder_test
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

filenames_Y = ["DoS_Y.txt"]
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

test_X = test_X.reshape(n_batches,nset,n_features)

normal = 0
attack = 0
for i in range(n_batches):
    z = np.sum(b[i*nset:i*nset+nset])#Here dif is not written as the messages we are testing from the attack datasets start from the index '0'. So it does not matter. No need to panic.
    if(z < 1):#Normal
        test_Y[i][0] = 0
        normal = normal + 1
    if(z >= 1):#Attack
        test_Y[i][0]= 1
        attack = attack + 1

model_input = Input(shape=(nset,n_features,1))
x = Conv2D(128,3,padding='same')(model_input)
x = Activation(activation='relu')(x)
x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
x = Conv2D(64,3,padding='same')(x)
x = Activation(activation='relu')(x)
encoded = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
x = Conv2DTranspose(64,3,strides=2,padding='same')(encoded) #There was an error here in the model. Encoded was not written maybe not the correct output.
x = Activation(activation='relu')(x)
x = Conv2DTranspose(128,3,strides=2,padding='same')(x)
x = Activation(activation='relu')(x)
x = Conv2D(1,3, padding='same')(x)#When the output value in not beetween 0 and 1 then don't use sigmoid. That will ruin the output of the whole model. This was the point that was earlier causing problems probably.
decoded = Activation(activation='sigmoid')(x)

#Training with Batchnormalization improved the model convergence but in testing with anomolous messages..those messages were also being classified as normal messages.

# model_input = Input(shape=(nset,n_features,1))
# x = Conv2D(256,3,padding='same')(model_input)
# x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# x = MaxPooling2D(pool_size = (1, 2), padding='same')(x)
# x = Conv2D(64,3,padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# encoded = MaxPooling2D(pool_size = (1, 2), padding='same')(x)
# x = Conv2D(64,3,padding='same')(encoded) #There was an error here in the model. Encoded was not written maybe not the correct output.
# x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# x = UpSampling2D((1, 2))(x)
# x = Conv2D(256,3,padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# x = UpSampling2D((1, 2))(x)
# decoded = Conv2D(1,3, activation='sigmoid', padding='same')(x)#When the output value in not beetween 0 and 1 then don't use sigmoid. That will ruin the output of the whole model. This was the point that was earlier causing problems probably.

# model_input = Input(shape=(nset*n_features))
# x = Dense(128)(model_input)
# # x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# # x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
# #x = Dropout(0.4)(x)
# x = Dense(32)(x)
# # x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# # encoded = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
# encoded = Dense(32)(x)
# # x = BatchNormalization()(x)
# x = Activation(activation='relu')(encoded)
# # x = UpSampling2D((2, 2))(x)
# x = Dense(128)(x)
# # x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# # x = UpSampling2D((2, 2))(x)
# decoded = Dense(40, activation='relu')(x)#When the output value in not beetween 0 and 1 then don't use sigmoid. That will ruin the output of the whole model. This was the point that was earlier causing problems probably.

model = tf.keras.Model(inputs=model_input, outputs=decoded, name="autoencoder_vitis")
model.summary()
# path = './autoencoder_conv2D_id_v5/weights.050.h5'
# path = './autoencoder_conv2D_id_v6/weights.050.h5'
# path = './autoencoder_conv2D_id_v7/weights.099.h5'
# path = './autoencoder_conv2D_id_v8/weights.092.h5'
# path = './autoencoder_conv2D_id_v9/weights.011.h5'
# path = './autoencoder_conv2D_id_v10/weights.072.h5'
# path = './autoencoder_conv2D_id_v11/weights.096.h5'#Trained only on ID's and block size 76.
# path = './autoencoder_conv2D_id_v12/weights.100.h5'#Trained only on ID's and block size 100.
path = './autoencoder_conv2D_id_v13/weights.080.h5'#Trained only on ID's and block size 100 with a separate sigmoid layer to make it compatible to Vitis-AI.

model.load_weights(path)
c = model.predict(test_X)
cc = c.round(decimals = 0)#Rounds values from 0.5 or less to 0 and 0.51 or more to 1
count = np.zeros(8) 
TN = 0
TP = 0
FN = 0
FP = 0
max_val = 0
threshold_const = 0
for j in range(40):
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
        if(distance <= threshold and test_Y[i] == 0):
            TN = TN + 1
        elif(distance > threshold and test_Y[i] == 1):
            TP = TP + 1
        elif(distance <= threshold and test_Y[i] == 1):
            FN = FN + 1
        elif(distance > threshold and test_Y[i] == 0):
            FP = FP + 1
# This is for normal message dataset
        # if(distance <= threshold):
        #     TN = TN + 1
        # elif(distance > threshold):
        #     FP = FP + 1

# This is for the normal datasets
    # print('Threshold '+str(threshold))
    # print('TN = '+str(TN))
    # print('FP = '+str(FP))
    # print('Accuracy = '+str(TN))
    # print('Max Error = '+str(max_val))


#This is for the attack datasets
    print('Threshold '+str(threshold))
    print('Normal messages = '+str(normal))
    print('Attack messages = '+str(attack))
    print('TN = '+str(TN))
    print('TP = '+str(TP))
    print('FN = '+str(FN))
    print('FP = '+str(FP))
    print('Accuracy = '+str(TN+TP))
    print('Max Error = '+str(max_val))
    print('----------------------------')

#Checking the round() function here.
    # check  = np.zeros(7)
    # check[0] = 0
    # check[1] = 0.25
    # check[2] = 0.5
    # check[3] = 0.75
    # check[4] = 0.99
    # check[5] = 0.51
    # check[6] = 0.49

    # print('--------------------------')

# check = check.round(decimals = 0)
# for o in range(len(check)):
#     print(check[o])

# print('--------------------------')
