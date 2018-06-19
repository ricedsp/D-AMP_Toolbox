__author__ = 'cmetzler&alimousavi'

import numpy as np
import argparse
import tensorflow as tf
import time
import LearnedDAMP as LDAMP
from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    type=bool,
    nargs="?",
    const=True,
    default=False,
    help="Use debugger to track down bad values during training")
parser.add_argument(
    "--alg",
    type=str,
    default="DAMP",#Options are DAMP, DIT, DVAMP, and DGAMP
    help="Which algorithm to use")
parser.add_argument(
    "--init_method",
    type=str,
    default="smaller_net",#Options are random, denoiser, smaller net, and layer_by_layer.
    #default='layer_by_layer',
    help="Which method to use for init")
parser.add_argument(
    "--start_layer",
    type=int,
    default=1,
    #default=5,
    help="Which layer(s) to start training")
parser.add_argument(
    "--train_end_to_end",
    type=bool,
    nargs="?",
    const=True,
    default=False,
    #default=True,
    help="Train end-to-end instead of layer-by-layer")
parser.add_argument(
    "--DnCNN_layers",
    type=int,
    default=16,
    help="How many DnCNN layers to use within each AMP layer")
parser.add_argument(
    "--tie_weights",
    type=bool,
    nargs="?",
    const=True,
    default=False,
    help="Use the same denoiser weights at every iteration")
parser.add_argument(
    "--loss_func",
    type=str,
    default="MSE",#Options are SURE, GSURE, or MSE
    help="Which loss function to use")
FLAGS, unparsed = parser.parse_known_args()

print(FLAGS)

## Network Parameters
alg=FLAGS.alg
tie_weights=FLAGS.tie_weights
height_img = 40
width_img = 40
channel_img = 1 # RGB -> 3, Grayscale -> 1
filter_height = 3
filter_width = 3
num_filters = 64
n_DnCNN_layers=FLAGS.DnCNN_layers
max_n_DAMP_layers=10#Unless FLAGS.start_layer is set to this value or LayerbyLayer=false, the code will sequentially train larger and larger networks end-to-end.

## Training Parameters
start_layer=FLAGS.start_layer
max_Epoch_Fails=3#How many training epochs to run without improvement in the validation error
ResumeTraining=False#Load weights from a network you've already trained a little
LayerbyLayer=not FLAGS.train_end_to_end #Train only the last layer of the network
if tie_weights==True:
    LayerbyLayer=False
    start_layer = max_n_DAMP_layers
learning_rates = [0.001, 0.0001]#, 0.00001]
EPOCHS = 50
n_Train_Images=128*1600#128*3000
n_Val_Images=10000#10000#Must be less than 21504
BATCH_SIZE = 128
InitWeightsMethod=FLAGS.init_method
if LayerbyLayer==False:
    BATCH_SIZE = 16
loss_func = FLAGS.loss_func

## Problem Parameters
sampling_rate=.2
sigma_w=1./255.#Noise std
n=channel_img*height_img*width_img
m=int(np.round(sampling_rate*n))
measurement_mode='gaussian'#'gaussian'#'coded-diffraction'#

# Parameters to to initalize weights. Won't be used if old weights are loaded
init_mu = 0
init_sigma = 0.1

train_start_time=time.time()
for n_DAMP_layers in range(start_layer,max_n_DAMP_layers+1,1):
    ## Clear all the old variables, tensors, etc.
    tf.reset_default_graph()

    LDAMP.SetNetworkParams(new_height_img=height_img, new_width_img=width_img, new_channel_img=channel_img, \
                           new_filter_height=filter_height, new_filter_width=filter_width, new_num_filters=num_filters, \
                           new_n_DnCNN_layers=n_DnCNN_layers, new_n_DAMP_layers=n_DAMP_layers,
                           new_sampling_rate=sampling_rate, \
                           new_BATCH_SIZE=BATCH_SIZE, new_sigma_w=sigma_w, new_n=n, new_m=m, new_training=True)
    LDAMP.ListNetworkParameters()

    # tf Graph input
    training_tf = tf.placeholder(tf.bool, name='training')
    x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

    ## Initialize the variable theta which stores the weights and biases
    if tie_weights == True:
        n_layers_trained = 1
    else:
        n_layers_trained = n_DAMP_layers
    theta = [None] * n_layers_trained
    for iter in range(n_layers_trained):
        with tf.variable_scope("Iter" + str(iter)):
            theta_thisIter = LDAMP.init_vars_DnCNN(init_mu, init_sigma)
        theta[iter] = theta_thisIter

    ## Construct the measurement model and handles/placeholders
    [A_handle, At_handle, A_val, A_val_tf] = LDAMP.GenerateMeasurementOperators(measurement_mode)
    y_measured = LDAMP.GenerateNoisyCSData_handles(x_true, A_handle, sigma_w, A_val_tf)

    ## Construct the reconstruction model
    if alg=='DAMP':
        (x_hat, MSE_history, NMSE_history, PSNR_history, r_final, rvar_final, div_overN) = LDAMP.LDAMP(y_measured,A_handle,At_handle,A_val_tf,theta,x_true,tie=tie_weights,training=training_tf,LayerbyLayer=LayerbyLayer)
    elif alg=='DIT':
        (x_hat, MSE_history, NMSE_history, PSNR_history) = LDAMP.LDIT(y_measured,A_handle,At_handle,A_val_tf,theta,x_true,tie=tie_weights,training=training_tf,LayerbyLayer=LayerbyLayer)
    else:
        raise ValueError('alg was not a supported option')

    ## Define loss and determine which variables to train
    nfp = np.float32(height_img * width_img)
    if loss_func=='SURE':
        assert alg=='DAMP', "Only LDAMP supports training with SURE"
        cost = LDAMP.MCSURE_loss(x_hat, div_overN, r_final, tf.sqrt(rvar_final))
    elif loss_func=='GSURE':
        assert alg == 'DAMP', "Only LDAMP currently supports training with GSURE"
        temp0=tf.matmul(A_val_tf,A_val_tf,transpose_b=True)
        temp1=tf.matrix_inverse(temp0)
        pinv_A=tf.matmul(A_val_tf,temp1,transpose_a=True)
        P=tf.matmul(pinv_A,A_val_tf)
        #Treat LDAMP/LDIT as a function of A^ty to calculate the divergence
        Aty_tf=At_handle(A_val_tf,y_measured)
        #Overwrite existing x_hat def
        (x_hat, _, _, _, _, _, _) = LDAMP.LDAMP_Aty(Aty_tf, A_handle,At_handle,A_val_tf, theta, x_true,tie=tie_weights,training=training_tf,LayerbyLayer=LayerbyLayer)
        if sigma_w==0.:#Not sure if TF is smart enough to avoid computing MCdiv when it doesn't have to
            MCdiv=0.
        else:
            #Calculate MC divergence of P*LDAMP(Aty)
            epsilon = tf.maximum(.001 * tf.reduce_max(Aty_tf, axis=0), .00001)
            eta = tf.random_normal(shape=Aty_tf.get_shape(), dtype=tf.float32)
            Aty_perturbed_tf=Aty_tf+tf.multiply(eta, epsilon)
            (x_hat_perturbed, _, _, _, _, _, _) = LDAMP.LDAMP_Aty(Aty_perturbed_tf, A_handle, At_handle, A_val_tf, theta, x_true,
                                                        tie=tie_weights,training=training_tf,LayerbyLayer=LayerbyLayer)
            Px_hat_perturbed=tf.matmul(P,x_hat_perturbed)
            Px_hat=tf.matmul(P,x_hat)
            eta_dx = tf.multiply(eta, Px_hat_perturbed - Px_hat)
            mean_eta_dx = tf.reduce_mean(eta_dx, axis=0)
            MCdiv= tf.divide(mean_eta_dx, epsilon)*n
        x_ML=tf.matmul(pinv_A,y_measured)
        cost = LDAMP.MCGSURE_loss(x_hat,x_ML,P,MCdiv,sigma_w)
        #Note: This cost is missing a ||Px||^2 term and so is expected to go negative
    else:
        cost = tf.nn.l2_loss(x_true - x_hat) * 1. / nfp

    iter = n_DAMP_layers - 1
    if LayerbyLayer==True:
        vars_to_train=[]#List of only the variables in the last layer.
        for l in range(0, n_DnCNN_layers):
            #vars_to_train.extend([theta[iter][0][l], theta[iter][1][l]])
            vars_to_train.extend([theta[iter][0][l]])
        for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, beta, and gamma
            gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
            beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
            var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
            mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
            gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
            beta = [v for v in tf.global_variables() if v.name == beta_name][0]
            moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
            moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
            vars_to_train.extend([gamma,beta,moving_variance,moving_mean])
    else:
        vars_to_train=tf.trainable_variables()

    LDAMP.CountParameters()

    ## Load and Preprocess Training Data
    train_images = np.load('./TrainingData/TrainingData_patch'+str(height_img)+'.npy')
    train_images=train_images[range(n_Train_Images),0,:,:]
    assert (len(train_images)>=n_Train_Images), "Requested too much training data"

    val_images = np.load('./TrainingData/ValidationData_patch'+str(height_img)+'.npy')
    val_images=val_images[:,0,:,:]
    assert (len(val_images)>=n_Val_Images), "Requested too much validation data"

    x_train = np.transpose(np.reshape(train_images, (-1, channel_img * height_img * width_img)))
    x_val = np.transpose(np.reshape(val_images, (-1, channel_img * height_img * width_img)))

    ## Train the Model
    for learning_rate in learning_rates:
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=vars_to_train)

        optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step. Allows us to update averages w/in BN
            optimizer = optimizer0.minimize(cost, var_list=vars_to_train)

        saver_best = tf.train.Saver()  # defaults to saving all variables
        saver_dict={}
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())#Seems to be necessary for the batch normalization layers for some reason.

            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            start_time = time.time()
            print("Load Initial Weights ...")
            if ResumeTraining or learning_rate!=learning_rates[0]:
                ##Load previous values for the weights
                saver_initvars_name_chckpt = LDAMP.GenLDAMPFilename(alg, tie_weights, LayerbyLayer,loss_func=loss_func) + ".ckpt"
                for iter in range(n_layers_trained):#Create a dictionary with all the variables except those associated with the optimizer.
                    for l in range(0, n_DnCNN_layers):
                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/w": theta[iter][0][l]})#,
                                           #"Iter" + str(iter) + "/l" + str(l) + "/b": theta[iter][1][l]})
                    for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                        gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                        beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                        var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                        mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                        gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                        beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                        moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                        moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/gamma": gamma})
                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/beta": beta})
                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance": moving_variance})
                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean": moving_mean})
                    saver_initvars = tf.train.Saver(saver_dict)
                    saver_initvars.restore(sess, saver_initvars_name_chckpt)
                    print("Loaded wieghts from %s" % saver_initvars_name_chckpt)
            else:
                ## Load initial values for the weights.
                # To do so, one associates each variable with a key (e.g. theta[iter][0][0] with l1/w_DnCNN) and loads the l1/w_DCNN weights that were trained on the denoiser
                # To confirm weights were actually loaded, run sess.run(theta[0][0][0][0][0])[0][0]) before and after this statement. (Requires running sess.run(tf.global_variables_initializer()) first
                if InitWeightsMethod == 'layer_by_layer':
                    #load the weights from an identical network that was trained layer-by-layer
                    saver_initvars_name_chckpt = LDAMP.GenLDAMPFilename(alg, tie_weights, LayerbyLayer=True,loss_func=loss_func) + ".ckpt"
                    for iter in range(
                            n_layers_trained):  # Create a dictionary with all the variables except those associated with the optimizer.
                        for l in range(0, n_DnCNN_layers):
                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/w": theta[iter][0][l]})#,
                                               #"Iter" + str(iter) + "/l" + str(l) + "/b": theta[iter][1][l]})
                        for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                            gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                            beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                            var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                            mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                            gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                            beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                            moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                            moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/gamma": gamma})
                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/beta": beta})
                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance": moving_variance})
                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean": moving_mean})
                        saver_initvars = tf.train.Saver(saver_dict)
                        saver_initvars.restore(sess, saver_initvars_name_chckpt)
                if InitWeightsMethod=='denoiser':
                    #load initial weights that were trained on a denoising problem
                    saver_initvars_name_chckpt=LDAMP.GenDnCNNFilename(300./255.)+".ckpt"
                    iter = 0
                    for l in range(0, n_DnCNN_layers):
                        saver_dict.update({"l" + str(l) + "/w": theta[iter][0][l]})#, "l" + str(l) + "/b": theta[iter][1][l]})
                    for l in range(1,n_DnCNN_layers-1):#Associate variance, means, and beta
                        gamma_name = "Iter"+str(iter)+"/l" + str(l) + "/BN/gamma:0"
                        beta_name="Iter"+str(iter)+"/l" + str(l) + "/BN/beta:0"
                        var_name="Iter"+str(iter)+"/l" + str(l) + "/BN/moving_variance:0"
                        mean_name="Iter"+str(iter)+"/l" + str(l) + "/BN/moving_mean:0"
                        gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                        beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                        moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                        moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                        saver_dict.update({"l" + str(l) + "/BN/gamma": gamma})
                        saver_dict.update({"l" + str(l) + "/BN/beta": beta})
                        saver_dict.update({"l" + str(l) + "/BN/moving_variance": moving_variance})
                        saver_dict.update({"l" + str(l) + "/BN/moving_mean": moving_mean})
                    saver_initvars = tf.train.Saver(saver_dict)
                    saver_initvars.restore(sess, saver_initvars_name_chckpt)
                elif InitWeightsMethod=='smaller_net' and n_DAMP_layers!=1:
                    #Initialize wieghts using a smaller network's weights
                    saver_initvars_name_chckpt = LDAMP.GenLDAMPFilename(alg, tie_weights, LayerbyLayer,
                                                                        n_DAMP_layer_override=n_DAMP_layers - 1,loss_func=loss_func) + ".ckpt"

                    #Load the first n-1 iterations weights from a previously learned network
                    for iter in range(n_DAMP_layers-1):
                        for l in range(0, n_DnCNN_layers):
                            saver_dict.update({"Iter"+str(iter)+"/l" + str(l) + "/w": theta[iter][0][l]})#, "Iter"+str(iter)+"/l" + str(l) + "/b": theta[iter][1][l]})
                        for l in range(1,n_DnCNN_layers-1):#Associate variance, means, and beta
                            gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                            beta_name="Iter"+str(iter)+"/l" + str(l) + "/BN/beta:0"
                            var_name="Iter"+str(iter)+"/l" + str(l) + "/BN/moving_variance:0"
                            mean_name="Iter"+str(iter)+"/l" + str(l) + "/BN/moving_mean:0"
                            gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                            beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                            moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                            moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/gamma": gamma})
                            saver_dict.update({"Iter"+str(iter)+"/l" + str(l) + "/BN/beta": beta})
                            saver_dict.update({"Iter"+str(iter)+"/l" + str(l) + "/BN/moving_variance": moving_variance})
                            saver_dict.update({"Iter"+str(iter)+"/l" + str(l) + "/BN/moving_mean": moving_mean})
                        saver_initvars = tf.train.Saver(saver_dict)
                        saver_initvars.restore(sess, saver_initvars_name_chckpt)

                    #Initialize the weights of layer n by using the weights from layer n-1
                    iter=n_DAMP_layers-1
                    saver_dict={}
                    for l in range(0, n_DnCNN_layers):
                        saver_dict.update({"Iter" + str(iter-1) + "/l" + str(l) + "/w": theta[iter][0][l]})#,"Iter" + str(iter-1) + "/l" + str(l) + "/b": theta[iter][1][l]})
                    for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                        gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                        beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                        var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                        mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                        gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                        beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                        moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                        moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                        saver_dict.update({"Iter" + str(iter-1) + "/l" + str(l) + "/BN/gamma": gamma})
                        saver_dict.update({"Iter"+str(iter-1)+"/l"+ str(l) + "/BN/beta": beta})
                        saver_dict.update({"Iter"+str(iter-1)+"/l"+ str(l) + "/BN/moving_variance": moving_variance})
                        saver_dict.update({"Iter"+str(iter-1)+"/l" + str(l) + "/BN/moving_mean": moving_mean})
                    saver_initvars = tf.train.Saver(saver_dict)
                    saver_initvars.restore(sess, saver_initvars_name_chckpt)
                else:
                    #use random weights. This will occur for 1 layer networks if set to use smaller_net initialization
                    pass
            time_taken = time.time() - start_time

            print("Training ...")
            print()
            if __name__ == '__main__':
                save_name = LDAMP.GenLDAMPFilename(alg, tie_weights, LayerbyLayer,loss_func=loss_func)
                save_name_chckpt = save_name + ".ckpt"
                val_values = []
                print("Initial Weights Validation Value:")
                rand_inds = np.random.choice(len(val_images), n_Val_Images, replace=False)
                start_time = time.time()
                for offset in range(0, n_Val_Images - BATCH_SIZE + 1, BATCH_SIZE):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                    end = offset + BATCH_SIZE

                    # Generate a new measurement matrix
                    A_val=LDAMP.GenerateMeasurementMatrix(measurement_mode)
                    batch_x_val = x_val[:, rand_inds[offset:end]]

                    # Run optimization. This will both generate compressive measurements and then recontruct from them.
                    loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, A_val_tf:A_val, training_tf:False})
                    val_values.append(loss_val)
                time_taken = time.time() - start_time
                print np.mean(val_values)
                if not LayerbyLayer:#For end-to-end training save the initial state so that LDAMP end-to-end doesn't diverge when using a high training rate
                    best_val_error = np.mean(val_values)
                    best_sess = sess
                    print "********************"
                    save_path = saver_best.save(best_sess, save_name_chckpt)
                    print("Initial session model saved in file: %s" % save_path)
                else:#For layerbylayer training don't save the initial state. With LDIT the initial validation error was often better than the validation error after training 1 epoch. This caused the network not to update and eventually diverge as it got longer and longer
                    best_val_error = np.inf
                failed_epochs=0
                for i in range(EPOCHS):
                    if failed_epochs>=max_Epoch_Fails:
                        break
                    train_values = []
                    print ("This Training iteration ...")
                    rand_inds=np.random.choice(len(train_images), n_Train_Images,replace=False)
                    start_time = time.time()
                    for offset in range(0, n_Train_Images-BATCH_SIZE+1, BATCH_SIZE):#Subtract batch size-1 to avoid errors when len(train_images) is not a multiple of the batch size
                        end = offset + BATCH_SIZE

                        # Generate a new measurement matrix
                        A_val = LDAMP.GenerateMeasurementMatrix(measurement_mode)
                        batch_x_train = x_train[:, rand_inds[offset:end]]

                        # Run optimization. This will both generate compressive measurements and then recontruct from them.
                        _, loss_val = sess.run([optimizer,cost], feed_dict={x_true: batch_x_train, A_val_tf:A_val, training_tf:True})#Feed dict names should match with the placeholders
                        train_values.append(loss_val)
                    time_taken = time.time() - start_time
                    print np.mean(train_values)
                    val_values = []
                    print("EPOCH ",i+1," Validation Value:" )
                    rand_inds = np.random.choice(len(val_images), n_Val_Images, replace=False)
                    start_time = time.time()
                    for offset in range(0, n_Val_Images-BATCH_SIZE+1, BATCH_SIZE):#Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                        end = offset + BATCH_SIZE

                        # Generate a new measurement matrix
                        A_val = LDAMP.GenerateMeasurementMatrix(measurement_mode)
                        batch_x_val = x_val[:, rand_inds[offset:end]]

                        # Run optimization. This will both generate compressive measurements and then recontruct from them.
                        loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, A_val_tf: A_val, training_tf:False})
                        val_values.append(loss_val)
                    time_taken = time.time() - start_time
                    print np.mean(val_values)
                    if(np.mean(val_values) < best_val_error):
                        failed_epochs=0
                        best_val_error = np.mean(val_values)
                        best_sess = sess
                        print "********************"
                        save_path = saver_best.save(best_sess, save_name_chckpt)
                        print("Best session model saved in file: %s" % save_path)
                    else:
                        failed_epochs=failed_epochs+1
                    print "********************"
                    total_train_time = time.time() - train_start_time
                    print("Training time so far: %.2f seconds" % total_train_time)

save_name_txt=save_name+ ".txt"
f= open(save_name_txt, 'wb')
f.write("Total Training Time ="+str(total_train_time))
f.close()
