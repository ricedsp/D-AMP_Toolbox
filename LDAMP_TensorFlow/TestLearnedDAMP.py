__author__ = 'cmetzler&alimousavi'
#Generate test data and use it to test the learned network

import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import LearnedDAMP as LDAMP
import random
import h5py

## Network Parameters
alg="DAMP"
tie_weights=False
height_img = 256
width_img = 256
channel_img = 1 # RGB -> 3, Grayscale -> 1
filter_height = 3
filter_width = 3
num_filters = 64
n_DnCNN_layers=16
n_DAMP_layers=10
TrainLoss='MSE'

## Training parameters (Selects which weights to use)
LayerbyLayer=True
DenoiserbyDenoiser=False#Overrides LayerbyLayer
if DenoiserbyDenoiser:
    LayerbyLayer=float('nan')

## Testing/Problem Parameters
BATCH_SIZE = 1#Using a batch size larger than 1 will hurt the denoiser by denoiser trained network because it will use an average noise level, rather than a noise level specific to each image
n_Test_Images = 5
sampling_rate_test=.2#The sampling rate used for testing
sampling_rate_train=.2#The sampling rate that was used for training
sigma_w=0.
n=channel_img*height_img*width_img
m=int(np.round(sampling_rate_test*n))
measurement_mode='Fast-JL'#'coded-diffraction'#'gaussian'#'complex-gaussian'#

# Parameters to to initalize weights. Won't be used if old weights are loaded
init_mu = 0
init_sigma = 0.1

random.seed(1)

LDAMP.SetNetworkParams(new_height_img=height_img, new_width_img=width_img, new_channel_img=channel_img, \
                       new_filter_height=filter_height, new_filter_width=filter_width, new_num_filters=num_filters, \
                       new_n_DnCNN_layers=n_DnCNN_layers, new_n_DAMP_layers=n_DAMP_layers,
                       new_sampling_rate=sampling_rate_test, \
                       new_BATCH_SIZE=BATCH_SIZE, new_sigma_w=sigma_w, new_n=n, new_m=m, new_training=False, use_adaptive_weights=DenoiserbyDenoiser)
LDAMP.ListNetworkParameters()

# tf Graph input
x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

#Create handles for the measurement operator
[A_handle, At_handle, A_val, A_val_tf]=LDAMP.GenerateMeasurementOperators(measurement_mode)

## Initialize the variable theta which stores the weights and biases
if tie_weights == True:
    theta = [None]
    with tf.variable_scope("Iter" + str(0)):
        theta_thisIter = LDAMP.init_vars_DnCNN(init_mu, init_sigma)
    theta[0] = theta_thisIter
elif DenoiserbyDenoiser:
    noise_min_stds = [0, 10, 20, 40, 60, 80, 100, 150, 300]#This is currently hardcoded within LearnedDAMP_functionhelper
    noise_max_stds = [10, 20, 40, 60, 80, 100, 150, 300, 500]  # This is currently hardcoded within LearnedDAMP_functionhelper
    theta = [None]*len(noise_min_stds)
    for noise_level in range(len(noise_min_stds)):
        with tf.variable_scope("Adaptive_NL"+str(noise_level)):
            theta[noise_level]= LDAMP.init_vars_DnCNN(init_mu, init_sigma)
else:
    n_layers_trained = n_DAMP_layers
    theta = [None] * n_layers_trained
    for iter in range(n_layers_trained):
        with tf.variable_scope("Iter" + str(iter)):
            theta_thisIter = LDAMP.init_vars_DnCNN(init_mu, init_sigma)
        theta[iter] = theta_thisIter

## Construct model
y_measured= LDAMP.GenerateNoisyCSData_handles(x_true, A_handle, sigma_w, A_val_tf)
if alg == 'DAMP':
    (x_hat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr) = LDAMP.LDAMP(y_measured, A_handle, At_handle, A_val_tf, theta, x_true, tie=tie_weights)
elif alg == 'DIT':
    (x_hat, MSE_history, NMSE_history, PSNR_history) = LDAMP.LDIT(y_measured, A_handle, At_handle, A_val_tf, theta, x_true, tie=tie_weights)
else:
    raise ValueError('alg was not a supported option')

## Load and Preprocess Test Data
if height_img>50:
    test_im_name = "./TrainingData/StandardTestData_" + str(height_img) + "Res.npy"
else:
    test_im_name = "./TrainingData/TestData_patch" + str(height_img) + ".npy"
test_images = np.load(test_im_name)
test_images=test_images[:,0,:,:]
assert (len(test_images)>=n_Test_Images), "Requested too much Test data"

x_test = np.transpose( np.reshape(test_images, (-1, height_img * width_img * channel_img)))

# with tf.Session() as sess:
#     y_test=sess.run(y_measured,feed_dict={x_true: x_test, A_val_tf: A_val})#All the batches will use the same measurement matrix

## Test the Model
saver = tf.train.Saver()  # defaults to saving all variables
saver_dict={}

with tf.Session() as sess:
    if tie_weights == 1: # Load weights from pretrained denoiser
        save_name = LDAMP.GenDnCNNFilename(80. / 255.) + ".ckpt"
        for l in range(0, n_DnCNN_layers):
            saver_dict.update({"l" + str(l) + "/w": theta[0][0][l]})#, "l" + str(l) + "/b": theta[0][1][l]})
        for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
            gamma_name = "Iter" + str(0) + "/l" + str(l) + "/BN/gamma:0"
            beta_name = "Iter" + str(0) + "/l" + str(l) + "/BN/beta:0"
            var_name = "Iter" + str(0) + "/l" + str(l) + "/BN/moving_variance:0"
            mean_name = "Iter" + str(0) + "/l" + str(l) + "/BN/moving_mean:0"
            gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
            beta = [v for v in tf.global_variables() if v.name == beta_name][0]
            moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
            moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
            saver_dict.update({"l" + str(l) + "/BN/gamma": gamma})
            saver_dict.update({"l" + str(l) + "/BN/beta": beta})
            saver_dict.update({"l" + str(l) + "/BN/moving_variance": moving_variance})
            saver_dict.update({"l" + str(l) + "/BN/moving_mean": moving_mean})
        saver_initvars = tf.train.Saver(saver_dict)
        saver_initvars.restore(sess, save_name)
    elif DenoiserbyDenoiser:
        for noise_level in range(len(noise_min_stds)):
            noise_min_std=noise_min_stds[noise_level]
            noise_max_std = noise_max_stds[noise_level]
            save_name = LDAMP.GenDnCNNFilename(noise_min_std/ 255.,noise_max_std/255.) + ".ckpt"
            for l in range(0, n_DnCNN_layers):
                saver_dict.update({"l" + str(l) + "/w": theta[noise_level][0][l]})#, "l" + str(l) + "/b": theta[noise_level][1][l]})
            for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                gamma_name = "Adaptive_NL"+str(noise_level) + "/l" + str(l) + "/BN/gamma:0"
                beta_name = "Adaptive_NL"+str(noise_level) + "/l" + str(l) + "/BN/beta:0"
                var_name = "Adaptive_NL"+str(noise_level) + "/l" + str(l) + "/BN/moving_variance:0"
                mean_name = "Adaptive_NL"+str(noise_level) + "/l" + str(l) + "/BN/moving_mean:0"
                gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                saver_dict.update({"l" + str(l) + "/BN/gamma": gamma})
                saver_dict.update({"l" + str(l) + "/BN/beta": beta})
                saver_dict.update({"l" + str(l) + "/BN/moving_variance": moving_variance})
                saver_dict.update({"l" + str(l) + "/BN/moving_mean": moving_mean})
            saver_initvars = tf.train.Saver(saver_dict)
            saver_initvars.restore(sess, save_name)
    else:
        #save_name = LDAMP.GenLDAMPFilename(alg, tie_weights, LayerbyLayer) + ".ckpt"
        save_name = LDAMP.GenLDAMPFilename(alg, tie_weights, LayerbyLayer,sampling_rate_override=sampling_rate_train,loss_func=TrainLoss) + ".ckpt"
        saver.restore(sess, save_name)

    print("Reconstructing Signal")
    start_time = time.time()

    Final_PSNRs=[]
    for offset in range(0, n_Test_Images - BATCH_SIZE + 1, BATCH_SIZE):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
        end = offset + BATCH_SIZE
        # batch_y_test = y_test[:, offset:end] #To be used when using precomputed measurements

        # Generate a new measurement matrix
        A_val = LDAMP.GenerateMeasurementMatrix(measurement_mode)

        batch_x_test = x_test[:, offset:end]

        # Run optimization. This will both generate compressive measurements and then recontruct from them.
        batch_x_recon, batch_MSE_hist, batch_NMSE_hist, batch_PSNR_hist = sess.run([x_hat, MSE_history, NMSE_history, PSNR_history], feed_dict={x_true: batch_x_test, A_val_tf: A_val})
        Final_PSNRs.append(batch_PSNR_hist[-1][0])
    print(Final_PSNRs)
    print(np.mean(Final_PSNRs))
    fig1 = plt.figure()
    plt.imshow(np.transpose(np.reshape(x_test[:, n_Test_Images-1], (height_img, width_img))), interpolation='nearest', cmap='gray')
    plt.show()
    fig2 = plt.figure()
    plt.imshow(np.transpose(np.reshape(batch_x_recon[:, 0], (height_img, width_img))), interpolation='nearest', cmap='gray')
    plt.show()
    fig3 = plt.figure()
    plt.plot(range(n_DAMP_layers+1), np.mean(batch_PSNR_hist,axis=1))
    plt.title("PSNR over " +str(alg)+" layers")
    plt.show()