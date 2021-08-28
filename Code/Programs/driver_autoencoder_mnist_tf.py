# driver_autoencoder_mnist_tf.py

import data_mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

# (1) Set up data
nsample = 60000
mnist = data_mnist.mnist()
# if issues loading dataset because of memory constraints on your machine
# set nsample = 10000 or fewer and use line X,_ = mnist.load_valid(nsample)
X,_ = mnist.load_train(nsample)
# plot 25 random digits from dataset
# set seed to be used when randomly picking images to plot
seed = 11
mnist.plot_image(X,seed)
# (2) Define model
# ndim = 784 dimensions for MNIST images
ndim = X.shape[0]
nreduced_dim = 87
# input encode and decode layers
input_layer_encoder = tf.keras.Input(shape=(ndim,))
code = tf.keras.layers.Dense(nreduced_dim, activation="linear", name="encoder")(input_layer_encoder)
reconstructed = tf.keras.layers.Dense(ndim, activation="linear", name="decoder")(code)
# create encoder and full autoencoder models
model_encoder = tf.keras.Model(inputs=input_layer_encoder, outputs=code)
model_autoencoder = tf.keras.Model(inputs=input_layer_encoder, outputs=reconstructed)
# set up loss function (mse = mean squared error) and optimizer for minimizing loss
loss_fun = "mse"
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
model_autoencoder.compile(optimizer=optimizer, loss="mse")
model_autoencoder.summary()
# (4) Train model
# nepoch is number of steps of optimization
nepoch = 10
time_start = time.time()
# tensorflow requires feature matrix to have sample axis along row so take transpose
history = model_autoencoder.fit(X.T,X.T, batch_size=128, epochs=nepoch)
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Generate and plot reconstructed data
# apply model_encoder predict method to transposed input to get coded result
# apply model_autoencoder predict method to transposed input to get reconstructed result
# transpose result so that sample axis is along columns so it can be plotted
coded_result = model_encoder.predict(X.T).T
print("Shape of coded_result: {}".format(coded_result.shape))
reconstructed_result = model_autoencoder.predict(X.T).T
mnist.plot_image(reconstructed_result,seed)
plt.show()