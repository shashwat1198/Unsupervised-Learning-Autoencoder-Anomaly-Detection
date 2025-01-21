import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation
from scipy.spatial.distance import hamming

# filenames_X = ["./dos_output.csv"]
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["./fuzzy_output.csv"]
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_X = ["./rpm_output.csv"]
# X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

filenames_X = ["./gear_output.csv"]
X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

# filenames_Y = ["DoS_Y.txt"]
# Y_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Y])

# filenames_Y = ["Fuzzy_Y.txt"]
# Y_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Y])

# filenames_Y = ["RPM_Y.txt"]
# Y_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Y])

filenames_Y = ["Gear_Y.txt"]
Y_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Y])

# filenames_Z = ["./dos_big_id.txt"]
# Z_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Z])

# filenames_Z = ["./fuzzy_big_id.txt"]
# Z_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Z])

# filenames_Z = ["./rpm_big_id.txt"]
# Z_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Z])

filenames_Z = ["./gear_big_id.txt"]
Z_set = np.concatenate([np.loadtxt(g,delimiter = ",") for g in filenames_Z])

a = np.array(X_set)
# print(a[0][0])
b = np.array(Y_set)
zzz = np.array(Z_set)
samples = 200000
dif = 0
nset = 100
n_features = 12
n_batches = int(samples/nset)
test_X = np.zeros([n_batches,nset,n_features])
test_Y = np.zeros([n_batches,1])
test_Z = np.zeros([n_batches,nset,n_features])

test_X = a.reshape([2000,100,12,1])

for i in range(n_batches):
    z = np.sum(b[i*nset+dif:i*nset+nset+dif])
    if(z < 1):#Normal
        test_Y[i][0] = 0
    if(z >= 1):#Attack
        test_Y[i][0] = 1

for i in range(n_batches):
    for j in range(nset):
        for k in range(n_features):#11
            test_Z[i][j][k] = zzz[i*nset+j+dif][k]

test_Z = test_Z.reshape(n_batches,nset,n_features,1)

model_input = (Input(shape=(100,12,1)))
y = Activation(activation='sigmoid')(model_input)

model = tf.keras.Model(inputs=model_input, outputs=y, name="autoencoder_test")
model.summary()

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
        y_true = test_Z[i].flatten()
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
    # print('Normal messages = '+str(normal))
    # print('Attack messages = '+str(attack))
    print('TN = '+str(TN))
    print('TP = '+str(TP))
    print('FN = '+str(FN))
    print('FP = '+str(FP))
    print('Accuracy = '+str(TN+TP))
    print('Max Error = '+str(max_val))
    print('----------------------------')