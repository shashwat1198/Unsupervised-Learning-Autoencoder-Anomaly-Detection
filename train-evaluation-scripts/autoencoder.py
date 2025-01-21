import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.keras import applications, Sequential, utils
from tensorflow.keras.layers import Reshape, Input, ReLU, UpSampling2D,UpSampling1D, Conv2DTranspose, Dense, BatchNormalization, Dropout , MaxPool1D, Flatten,Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import L1L2

#Precision, recall, f1_score computation
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
import os
import matplotlib.pyplot as plt

# filenames_X = ["normal_id_data_train.csv"]
filenames_X = ["autoencoder_id_train.txt"]#The autoencoder is only trained on normal CAN messages. With ID's only.
# filenames_X = ["normal_id_data_binary_train.csv"]
X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

a = np.array(X_set)
print(a.shape)

#Train labels
samples= 850000
# Hyperparameters
nset = 100#76
n_features =12#76 #10
batch_size = 64
n_batches = int(samples/nset)
epochs = 100
validation_split = 0.15
checkpoint_dir = 'autoencoder_conv2D_id_v13'#TL_spoof_comb #autoencoder_conv2D_id_v5

train_X = np.zeros([n_batches,nset,n_features])#11

for i in range(n_batches):
    for j in range(nset):
        for k in range(n_features):#11
            train_X[i][j][k] = a[i*nset+j][k]

train_X = train_X.reshape(n_batches,nset,n_features)
# train_X = train_X.reshape(samples,48,1)
train_X.shape

model_input = Input(shape=(nset,n_features,1))
x = Conv2D(128,3,padding='same')(model_input)
# x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
x = Conv2D(64,3,padding='same')(x)
# x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
encoded = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
x = Conv2DTranspose(64,3,strides=2,padding='same')(encoded) #There was an error here in the model. Encoded was not written maybe not the correct output.
# x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x = Conv2DTranspose(128,3,strides=2,padding='same')(x)
# x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x = Conv2D(1,3,padding='same')(x)
decoded = Activation(activation='sigmoid')(x)#When the output value in not beetween 0 and 1 then don't use sigmoid. That will ruin the output of the whole model. This was the point that was earlier causing problems probably.

#Training with Batchnormalization improved the model convergence but in testing with anomolous messages..those messages were also being classified as normal messages.

# model_input = Input(shape=(48,1))
# x = Dense(256)(model_input)
# x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# x = MaxPooling1D(pool_size=2,strides=2, padding='valid')(x)
# x = Dense(64)(x)
# x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# encoded = MaxPooling1D(pool_size=2,strides=2, padding='valid')(x)
# x = Dense(64)(encoded) #There was an error here in the model. Encoded was not written maybe not the correct output.
# x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# x = UpSampling1D(2)(x)
# x = Dense(256)(x)
# x = BatchNormalization()(x)
# x = Activation(activation='relu')(x)
# x = UpSampling1D(2)(x)
# decoded = Dense(48, activation='sigmoid')(x)#When the output value in not beetween 0 and 1 then don't use sigmoid. That will ruin the output of the whole model. This was the point that was earlier causing problems probably.

#Identify the encoder and the decoder part in the above model. There is only encoder part here : Done

model = tf.keras.Model(inputs=model_input, outputs=decoded, name="dense_model")
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())#tf.keras.losses.MeanSquaredError(), metrics=['accuracy',tf.keras.metrics.AUC() 'binary_crossentropy']#MeanSquaredError
os.mkdir(checkpoint_dir)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath =checkpoint_dir + '/weights.{epoch:03d}.h5',
        verbose = 1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto')
earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 50,
        mode='auto',
        verbose = 1)
history = model.fit(train_X, train_X, batch_size=batch_size, epochs=epochs,validation_split=validation_split,callbacks = [ cp_callback,earlystopping_callback],shuffle=True)