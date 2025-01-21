import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, ReLU, Conv2DTranspose, Conv2D, MaxPooling2D, Activation

filenames_X = ["./autoencoder_id_train.txt"]
X_set = np.concatenate([np.loadtxt(f,delimiter = ",") for f in filenames_X])

a = np.array(X_set)
print(a.shape)

#Calibration labels for the Vitis-AI model : During quantisation the model needs some unlabelled inputs from the training set to ensure correct calibration of the weights for the generated quantised model.
samples= 100000
# Hyperparameters
nset = 100#76
n_features =12#76 #10
n_batches = int(samples/nset)

train_X = np.zeros([n_batches,nset,n_features])#11
for i in range(n_batches):
    for j in range(nset):
        for k in range(n_features):
            train_X[i][j][k] = a[i*nset+j][k]

train_X = train_X.reshape(n_batches,nset,n_features)
print(train_X.shape)

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
x = Conv2D(1,3,padding='same')(x)
decoded = Activation(activation='sigmoid')(x)#When the output value in not beetween 0 and 1 then don't use sigmoid. That will ruin the output of the whole model. This was the point that was earlier causing problems probably.

float_model = tf.keras.Model(inputs=model_input, outputs=decoded, name="autoencoder_model")
path = './weights.080.h5'
float_model.load_weights(path)
float_model.summary()

#Quantizing the model step.
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(calib_dataset=train_X)#fold_conv_bn=True
#quantized_model = quantizer.quantize_model( calib_dataset=train_X, fold_conv_bn=False, fold_bn=False, replace_relu6=False, include_cle=True, cle_steps=10)
quantized_model.save('./quantised_cae_v3_100.h5')

from tensorflow_model_optimization.quantization.keras import vitis_quantize
with vitis_quantize.quantize_scope():
    model = tf.keras.models.load_model('./quantised_cae_v3_100.h5')
model.summary()
