# ##################################################################
# ############################ DC-GAN ##############################
# ##################################################################

# @author: d3vvy
# @date: 25.02.2017

# ##################################################################
# ##################### Imports and Setting ########################
# ##################################################################
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random 					
import os
import os.path
import sys

from functions.auxiliary_functions import arg_parser
from functions.auxiliary_functions import get_highest_model
from functions.auxiliary_functions import merge_and_save
from functions.celeba_input import CelebA

# ####################################
# ######## Training Parameters #######
# ####################################
Z_size = 100
batch_size = 64

# ####################################
# ############ Directories ########### 
# ####################################
# Path, where to save sample-images
sample_dir = '/'.join(os.path.realpath(__file__).split("/")[:-1])+'/save/figs'
if not os.path.exists(sample_dir):
	os.makedirs(sample_dir)

# Path, where to save trained model
model_dir = '/'.join(os.path.realpath(__file__).split("/")[:-1])+'/save/models'
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

# ####################################
# ######## Retrieve Arguments ########
# ####################################

arg_dict = arg_parser(sys.argv)

# should we train?
train = arg_dict["train"]

# should we load weight?
restore_weights = arg_dict["load"][0]

if restore_weights:
	# in case the value is not the deafult 0 anymore
	if arg_dict["load"][1] != 0:
		# set globa_step to the provided number
		global_step = arg_dict["load"][1]
	# in case it still is 0 and no nr was provided
	else:
		#get the one with the "highest number"
		global_step = get_highest_model(model_dir)
else:
	# in case no loading is requested, start from scratch
	global_step = 0

# how many batches should we do
if arg_dict["nbatch"][0] == False:
	#default
	n_batches = 2000
else:
	n_batches = arg_dict["nbatch"][1]

# visualisation turned on?
save_fig = arg_dict["vis"]

# should we do test/ z-interpolation/ z-change
z_inter = arg_dict["z_int"]

# set z_change and its parameter. If no number of paramter was
# requested, its a random it between 0 and 99 (bc of array)
z_change = arg_dict["z_change"][0]
z_param = arg_dict["z_change"][1]

# check if someone wants the test mode
test = arg_dict["test"]
if test:
	# start from highest number model
	test_load = get_highest_model(model_dir)

# set test_batch_size (by defualt 1)
test_batch_size = arg_dict["number"]

# ####################################
# ####### Print&Save-Settings ########
# ####################################
printFreq = np.round(2*1000/batch_size)
sampleFreq = np.round(10*1000/batch_size)
saveFreq = np.round(20*1000/batch_size)


# ####################################
# ### Creation of Sample-Supplier ####
# ####################################
sampleGen = CelebA()

# ##################################################################
# ##################### Auxiliary Functions ########################
# ##################################################################

# leaky rectified linear unit
def lrelu(X, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		foo1 = 0.5 * (1 + leak)
		foo2 = 0.5 * (1 - leak)
		return foo1 * X + foo2 * abs(X)

# determine learning rate update given fake and real accuracy
def check_LR(fake_acc, real_acc, D_rate, G_rate, initial_lr, thresh1=0.7, thresh2=0.5):
	if fake_acc < thresh2 or real_acc < thresh2:
		D_rate += 0.00001
		G_rate *= 0.8
	elif fake_acc > thresh1 or real_acc > thresh1: 
		D_rate *= 0.8
		G_rate += 0.00001

	D_rate = max(min(0.00025, D_rate), 0.00015)
	G_rate = max(min(0.00025, G_rate), 0.00015)
	return np.array([D_rate, G_rate])

# ####################################
# #### Function for Dense_Layers #####
# ####################################
def dense_layer(layer_input, W_shape, b_shape=[-1], activation=tf.nn.tanh, bias_init=0.1, batch_norm=False, norm_before_act=False, reuse=False, varscope=None, namescope=None):
	with tf.name_scope(namescope):
		with tf.variable_scope(varscope, reuse=reuse):
			if b_shape == [-1]:
				b_shape = [W_shape[-1]]
			W = tf.get_variable('weights', W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
			b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
			iState = tf.matmul(layer_input, W)
			if b_shape != [0]:
				iState += b
			if batch_norm and norm_before_act:
				iState = tf.nn.l2_normalize(iState, 0)
			iState = activation(iState)
			if batch_norm and not norm_before_act:
				iState = tf.nn.l2_normalize(iState, 0)
			return iState

# ####################################
# #### Function for Convolution ######
# ####################################
def conv2d_layer(layer_input, W_shape, b_shape=[-1], strides=[1,1,1,1], padding='SAME', activation=tf.nn.relu, bias_init=0.1, batch_norm=False, reuse=False, varscope=None, namescope=None):
	with tf.name_scope(namescope):
		with tf.variable_scope(varscope, reuse=reuse):
			if b_shape == [-1]:
				b_shape = [W_shape[-1]]
			W = tf.get_variable('weights', W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
			b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
			iState = tf.nn.conv2d(layer_input, W, strides, padding)
			if b_shape != [0]:
				iState += b
			if batch_norm:
				iState = tf.nn.l2_normalize(iState, 0)
			return activation(iState)

# ####################################
# #### Function for Trans-Conv ######
# ####################################
def trans2d_layer(layer_input, W_shape, output_shape, b_shape=[-2], strides=[1,1,1,1], padding='SAME', activation=tf.nn.relu, bias_init=0.1, batch_norm=False, reuse=False, varscope=None, namescope=None):
	with tf.name_scope(namescope):
		with tf.variable_scope(varscope, reuse=reuse):
			if b_shape == [-2]:
				b_shape = [W_shape[-2]]
			W = tf.get_variable('weights', W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
			b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
			iState = tf.nn.conv2d_transpose(layer_input, W, output_shape, strides)
			if b_shape != [0]:
				iState += b
			if batch_norm:
				iState = tf.nn.l2_normalize(iState, 0)
			return activation(iState)

# ####################################
# ######### Generator Setup ##########
# ####################################
def generator(Z, reuse=False):
	state = dense_layer(Z, W_shape=[Z_size,512*4*4], activation=tf.nn.relu, batch_norm=True, norm_before_act=True, reuse=reuse, varscope='g_fc1', namescope='generator')
	state = tf.reshape(state, [batch_size,4,4,512])
	state = trans2d_layer(layer_input=state, W_shape=[5,5,256,512], output_shape=[batch_size,8,8,256], strides=[1,2,2,1], batch_norm=True, reuse=reuse, varscope='g_trans1', namescope='generator')
	state = trans2d_layer(layer_input=state, W_shape=[5,5,128,256], output_shape=[batch_size,16,16,128], strides=[1,2,2,1], batch_norm=True, reuse=reuse, varscope='g_trans2',namescope='generator')
	# state = tf.nn.dropout(state,drop[1])
	state = trans2d_layer(layer_input=state, W_shape=[5,5,64,128], output_shape=[batch_size,32,32,64], strides=[1,2,2,1], batch_norm=True, reuse=reuse, varscope='g_trans3',namescope='generator')
	state = trans2d_layer(layer_input=state, W_shape=[5,5,1,64], b_shape=[0], output_shape=[batch_size,64,64,1], strides=[1,2,2,1], activation=tf.nn.tanh, batch_norm=False, reuse=reuse, varscope='g_trans5',namescope='generator')
	return state

# ####################################
# ####### Distinguisher Setup ########
# ####################################
def distinguisher(X, reuse=False):
	state = conv2d_layer(layer_input=X, W_shape=[5,5,1,128], b_shape=[0], strides=[1,2,2,1], activation=lrelu, reuse=reuse, varscope='d_conv1', namescope='distinguisher') # ==> ?
	state = conv2d_layer(layer_input=state, W_shape=[5,5,128,128], strides=[1,2,2,1], activation=lrelu, batch_norm=False, reuse=reuse, varscope='d_conv2', namescope='distinguisher') # ==> ?
	state = tf.nn.dropout(state,0.5)
	state = conv2d_layer(layer_input=state, W_shape=[5,5,128,128], strides=[1,2,2,1], activation=lrelu, batch_norm=False, reuse=reuse, varscope='d_conv3', namescope='distinguisher') 
	state = conv2d_layer(layer_input=state, W_shape=[5,5,128,128], strides=[1,2,2,1], activation=lrelu, batch_norm=False, reuse=reuse, varscope='d_conv4', namescope='distinguisher') 
	state = tf.reshape(state, [batch_size,128*4*4])
	state = tf.nn.dropout(state,0.5)
	state = dense_layer(state, [128*4*4,1], activation=tf.nn.sigmoid, reuse=reuse, varscope='d_fc2', namescope='distinguisher')
	return state

# ##################################################################
# ################ Definition of Data Flow Graph ###################
# ##################################################################
tf.reset_default_graph()
initializer = tf.truncated_normal_initializer(stddev=0.02)

# # define placeholders for input
Z = tf.placeholder(dtype=tf.float32, shape=[None,Z_size])
X = tf.placeholder(dtype=tf.float32, shape=[None,64,64,1])
lr = tf.placeholder(dtype=tf.float32,shape=[2])

# define instances of data flow
Gz = generator(Z) # generates images from Z
Dx = distinguisher(X) # produces probabilities for real images
Dg = distinguisher(Gz, reuse=True) # produces probabilities for generator images

# ####################################
# ########## Training Setup ##########
# ####################################
D_lossfun = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(Dx), labels=tf.ones(batch_size,1)) \
						 + tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(Dg), labels=tf.zeros(batch_size,1)))
G_lossfun = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(Dg), labels=tf.ones(batch_size,1)))

#G_perf = tf.reduce_mean(tf.equal(tf.round(Dg),tf.ones(batch_size,1)))
Dx_perf = tf.reduce_mean(tf.cast(tf.equal(tf.round(Dx),tf.ones(batch_size,1)),"float"))
Dg_perf = tf.reduce_mean(tf.cast(tf.equal(tf.round(Dg),tf.zeros(batch_size,1)),"float"))

# define which variables to optimize
train_vars = tf.trainable_variables()
D_vars = [var for var in train_vars if 'd_' in var.name]
G_vars = [var for var in train_vars if 'g_' in var.name]

# define optimizer
D_optimizer = tf.train.AdamOptimizer(learning_rate=lr[0],beta1=0.5)
G_optimizer = tf.train.AdamOptimizer(learning_rate=lr[1],beta1=0.5)
 
# compute gradients
D_gradients = D_optimizer.compute_gradients(D_lossfun,D_vars) #Only update the weights for the distinguisher network.
G_gradients = G_optimizer.compute_gradients(G_lossfun,G_vars) #Only update the weights for the generator network.

# apply gradients
D_update = D_optimizer.apply_gradients(D_gradients)
G_update = G_optimizer.apply_gradients(G_gradients)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1.0, write_version=tf.train.SaverDef.V1)

# ##################################################################
# ##################### Training of Network ########################
# ##################################################################

if train:
	with tf.Session() as sess: 

		print('\n\n\n-------------------------------------------')
		print('--------- [*] Training started... ---------')
		print('-------------------------------------------\n\n\n')

		sess.run(init)
		if restore_weights:
			# model name needs to be changed manually
			if os.path.isfile(model_dir+'/model.ckpt-' + str(global_step)): 
				saver.restore(sess, model_dir+'/model.ckpt-' + str(global_step))
				print('\n==> Model loaded from /model.ckpt-' + str(global_step) + '\n')
		
		# For later plotting
		G_loss_history = []
		D_loss_history = []

		D_fake_history = []
		D_real_history = []

		D_lr_history = []
		G_lr_history = []

		# first discriminator learningrate, second generative learningrate
		init_lr = np.array([0.0002,0.0002])
		learn_rates = init_lr

		# generate z_sample for progress visualization (does not change during training)
		np.random.seed(1)
		z_sample = np.random.uniform(-1.0,1.0, size=[batch_size,Z_size]).astype(np.float32) 

		for i in range(n_batches):
			# create inputs
			z_in = np.random.uniform(-1.0, 1.0, size=[batch_size,Z_size]).astype(np.float32)
			x_in = sampleGen.get_batch(batch_size)

			# update distinguisher 
			_,D_loss = sess.run([D_update,D_lossfun],feed_dict={Z: z_in, X: x_in, lr: learn_rates})

			# update generator
			_,G_loss = sess.run([G_update,G_lossfun],feed_dict={Z: z_in, lr: learn_rates})
			z_in = np.random.uniform(-1.0, 1.0, size=[batch_size,Z_size]).astype(np.float32)
			_,G_loss = sess.run([G_update,G_lossfun],feed_dict={Z: z_in, lr: learn_rates})

			# update learning rates given distinguisher performance
			D_fake,D_real = sess.run([Dg_perf,Dx_perf],feed_dict={Z: z_in, X: x_in})
			learn_rates = check_LR(D_fake,D_real,learn_rates[0],learn_rates[1],init_lr)

			# print progress after n=printFreq batches
			if (i+1)%printFreq == 0 or (i+1) == n_batches:
				print('--------Stats for Batch: ' + str(i+1) + '/' + str(n_batches) + ' --------')
				print('\tGenerator Loss:\t\t' + str(G_loss))
				print('\tDiscriminator Loss:\t' + str(D_loss))
				print('\tGenerator Learning Rate:\t%f' % (learn_rates[1]))
				print('\tDiscriminator Learning Rate:\t%f' % (learn_rates[0]))
				print('\tDiscriminator - Percentage of True Positives:\t' + str(D_real))
				print('\tDiscriminator - Percentage of False Negatives:\t' + str(D_fake) + '\n')


				G_loss_history.append(G_loss)
				D_loss_history.append(D_loss)

				D_fake_history.append(D_fake)
				D_real_history.append(D_real)

				D_lr_history.append(learn_rates[0])
				G_lr_history.append(learn_rates[1])

			# save samples documenting progress after n=sampleFreq batches
			if save_fig: and ((i+1)%sampleFreq == 0 or (i+1) == n_batches):
				# use z_sample to get sample images from generator
				Gz_sample = sess.run(Gz, feed_dict={Z: z_sample}) 
				merge_and_save(np.reshape(Gz_sample[0:25],[25,64,64]),[5,5],sample_dir+'/fig'+str(i+global_step)+'.png')

			# save model weights after n=saveFreq batches
			if (i+1)%saveFreq == 0 or (i+1) == n_batches:
				saver.save(sess,model_dir+'/model.ckpt', global_step=global_step+i)
				print(' ==> Model saved with number: '+str(global_step+i))


	print('\n\n\n-------------------------------------------')
	print('---------- [+] Training finished! ---------')
	print('-------------------------------------------\n\n\n')




	# ##################################################################
	# ######################## Training Plots ##########################
	# ##################################################################


	plt.figure()
	plt.subplot(311)
	plt.title('Network Losses')
	plt.xlabel('training progress')
	plt.plot(np.array(G_loss_history),label="gen loss")
	plt.plot(np.array(D_loss_history),label="dis loss")
	plt.legend()
	plt.subplot(312)
	plt.title('Discriminator Accuracy')
	plt.xlabel('training progress')
	plt.ylabel('percentage')
	plt.plot(np.array(D_fake_history),label="fake acc")
	plt.plot(np.array(D_real_history),label="real acc")
	plt.legend()
	plt.subplot(313)
	plt.title('Network Learning Rates')
	plt.xlabel('training progress')
	plt.plot(np.array(D_lr_history),label="dis lr")
	plt.plot(np.array(G_lr_history),label="gen lr")
	plt.legend()
	plt.tight_layout()

	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	plt.savefig(sample_dir+'/0_stats.png')


# ##################################################################
# ###################### Testing of Network ########################
# ##################################################################

if test:

	with tf.Session() as sess:

		print('\n\n\n-------------------------------------------')
		print('---------- [*] Testing started... ---------')
		print('-------------------------------------------\n\n\n')

		sess.run(init)

		# restore latest weights
		saver.restore(sess, model_dir+'/model.ckpt-'+ str(test_load))
		print('\n==> Model loaded from /model.ckpt-' + str(test_load) + '\n')
		if z_change:
			print('-------- [*] Z-Parameter Change in progress... --------')
			n_imgs = 4
			n_inter = 12
			inter_imgs = []

			for i in range(n_imgs):
				z_basis = np.random.uniform(-1.0,1.0, size=[1,Z_size]).astype(np.float32)
				sample = np.squeeze(np.array(batch_size*[z_basis]))
				for j in range(batch_size):
					sample[j][z_param] = -0.05 + j * (1/batch_size)

				Gz_sample = np.squeeze(sess.run(Gz, feed_dict={Z: sample}))
				inter_imgs.extend(Gz_sample[::int(batch_size/n_inter)])

			if not os.path.exists(sample_dir):
				os.makedirs(sample_dir)
			filename = sample_dir+'/Z-Change_param'+str(z_param+1)+'.png'

			merge_and_save(np.reshape(inter_imgs[0:n_imgs*n_inter],[n_imgs*n_inter,64,64]),[n_imgs,n_inter],filename)
			
			print('Saved to: ' + filename)
			print('-------- [+] Z-Parameter Change in finished! --------')

		# Z-Interpolation
		if z_inter:
			print('-------- [*] Z-Interpolation in progress... --------')

			# generate a 2 z-vectors
			z_value1 = np.random.uniform(-1.0,1.0, size=[1,Z_size]).astype(np.float32)
			z_value2 = np.random.uniform(-1.0,1.0, size=[1,Z_size]).astype(np.float32)
			z_ite = z_value1
			
			# create batch, first entry = first z-vec, last entry = seconz z-vec
			z_vec = np.zeros([batch_size,Z_size])
			z_vec[0] = z_value1
			z_vec[batch_size-1] = z_value2

			# calculate increment
			increment = (z_value2-z_value1)/(batch_size-1)

			# apply increment and zave intepolating z-vectors
			for counter in range(1,batch_size-1):
				z_ite += increment
				z_vec[counter] = z_ite

			# generate picture
			sample = np.squeeze(np.array([z_vec]))
			Gz_sample = np.squeeze(sess.run(Gz, feed_dict={Z: sample}))

			# save
			if not os.path.exists(sample_dir):
				os.makedirs(sample_dir)
			filename = sample_dir+'/Z-interpolation.png'

			merge_and_save(np.reshape(Gz_sample[0:batch_size],[batch_size,64,64]),[int(np.sqrt(batch_size)),int(np.sqrt(batch_size))],filename)
			
			print('Saved to: ' + filename)
			print('-------- [+] Z-Interpolation finished! --------')
		
		if not (z_inter or z_change):
			print('\n---------- [*] Image Generation in progress... ----------')
			# generate batch of z-vectors and feed through the generator
			z_sample = np.random.uniform(-1.0,1.0, size=[batch_size,Z_size]).astype(np.float32)
			Gz_sample = sess.run(Gz, feed_dict={Z: z_sample})
			
			#save it
			if not os.path.exists(sample_dir):
				os.makedirs(sample_dir)
			filename = sample_dir+'/'+str(test_batch_size)+'x'+str(test_batch_size)+'_sample.png'
			merge_and_save(np.reshape(Gz_sample[0:test_batch_size**2],[test_batch_size**2,64,64]),[test_batch_size,test_batch_size],filename)
			
			print('Saved to: ' + filename)
			print('---------- [*] Image Generation finished! ----------:\n')

	print('\n\n\n-------------------------------------------')
	print('---------- [+] Testing finished! ----------')
	print('-------------------------------------------\n\n\n')
