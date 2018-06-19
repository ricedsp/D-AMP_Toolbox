__author__ = 'cmetzler&alimousavi'

import numpy as np
import argparse
import tensorflow as tf
import time
import LearnedDAMP as LDAMP
from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt
import h5py

## Network Parameters
height_img = 256
width_img = 256
channel_img = 1 # RGB -> 3, Grayscale -> 1
filter_height = 3
filter_width = 3
num_filters = 64
n_DnCNN_layers=16
useSURE=False#Use the network trained with ground-truth data or with SURE


## Training Parameters
BATCH_SIZE = 1

## Problem Parameters
sigma_w=25./255.#Noise std
n=channel_img*height_img*width_img

# Parameters to to initalize weights. Won't be used if old weights are loaded
init_mu = 0
init_sigma = 0.1

train_start_time=time.time()

## Clear all the old variables, tensors, etc.
tf.reset_default_graph()

LDAMP.SetNetworkParams(new_height_img=height_img, new_width_img=width_img, new_channel_img=channel_img, \
                       new_filter_height=filter_height, new_filter_width=filter_width, new_num_filters=num_filters, \
                       new_n_DnCNN_layers=n_DnCNN_layers, new_n_DAMP_layers=None,
                       new_sampling_rate=None, \
                       new_BATCH_SIZE=BATCH_SIZE, new_sigma_w=sigma_w, new_n=n, new_m=None, new_training=False)
LDAMP.ListNetworkParameters()

# tf Graph input
x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

## Construct the measurement model and handles/placeholders
y_measured = LDAMP.AddNoise(x_true,sigma_w)

## Initialize the variable theta which stores the weights and biases
theta_dncnn=LDAMP.init_vars_DnCNN(init_mu, init_sigma)

## Construct the reconstruction model
x_hat = LDAMP.DnCNN(y_measured,None,theta_dncnn,training=False)

LDAMP.CountParameters()

## Load and Preprocess Test Data
if height_img>50:
    test_im_name = "./TrainingData/StandardTestData_" + str(height_img) + "Res.npy"
else:
    test_im_name = "./TrainingData/TestData_patch" + str(height_img) + ".npy"

test_images = np.load(test_im_name)
test_images=test_images[:,0,:,:]
assert (len(test_images)>=BATCH_SIZE), "Requested too much Test data"

x_test = np.transpose( np.reshape(test_images[0:BATCH_SIZE], (BATCH_SIZE, height_img * width_img * channel_img)))
with tf.Session() as sess:
    y_test=sess.run(y_measured,feed_dict={x_true: x_test})

## Train the Model
saver = tf.train.Saver()  # defaults to saving all variables
saver_dict={}
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # if 255.*sigma_w<10.:
    #     sigma_w_min=0.
    #     sigma_w_max=10.
    # elif 255.*sigma_w<20.:
    #     sigma_w_min=10.
    #     sigma_w_max=20.
    # elif 255.*sigma_w < 40.:
    #     sigma_w_min = 20.
    #     sigma_w_max = 40.
    # elif 255.*sigma_w < 60.:
    #     sigma_w_min = 40.
    #     sigma_w_max = 60.
    # elif 255.*sigma_w < 80.:
    #     sigma_w_min = 60.
    #     sigma_w_max = 80.
    # elif 255.*sigma_w < 100.:
    #     sigma_w_min = 80.
    #     sigma_w_max = 100.
    # elif 255.*sigma_w < 150.:
    #     sigma_w_min = 100.
    #     sigma_w_max = 150.
    # elif 255.*sigma_w < 300.:
    #     sigma_w_min = 150.
    #     sigma_w_max = 300.
    # else:
    #     sigma_w_min = 300.
    #     sigma_w_max = 500.
    sigma_w_min=sigma_w*255.
    sigma_w_max=sigma_w*255.

    save_name = LDAMP.GenDnCNNFilename(sigma_w_min/255.,sigma_w_max/255.,useSURE=useSURE)
    save_name_chckpt = save_name + ".ckpt"
    saver.restore(sess, save_name_chckpt)

    print("Reconstructing Signal")
    start_time = time.time()
    [reconstructed_test_images]= sess.run([x_hat], feed_dict={y_measured: y_test})
    time_taken=time.time()-start_time
    fig1 = plt.figure()
    plt.imshow(np.transpose(np.reshape(x_test[:, 0], (height_img, width_img))), interpolation='nearest', cmap='gray')
    plt.show()
    fig2 = plt.figure()
    plt.imshow(np.transpose(np.reshape(y_test[:, 0], (height_img, width_img))), interpolation='nearest', cmap='gray')
    plt.show()
    fig3 = plt.figure()
    plt.imshow(np.transpose(np.reshape(reconstructed_test_images[:, 0], (height_img, width_img))), interpolation='nearest', cmap='gray')
    plt.show()
    [_,_,PSNR]=LDAMP.EvalError_np(x_test[:, 0],reconstructed_test_images[:, 0])
    print(PSNR)
    1