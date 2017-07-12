import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models

def predict(model_data_path, image_path):

	# Default input size
	height = 228
	width = 304
	channels = 3
	batch_size = 1
	
	# Read image
	img = Image.open(image_path)
	img = img.resize([width,height], Image.ANTIALIAS)
	img = np.array(img).astype('float32')
	img = np.expand_dims(np.asarray(img), axis = 0)
	
	print('Constructing the network...')
	input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels), name='input')
	net = models.ResNet50UpProj({'data': input_node}, batch_size)
		
	with tf.Session() as sess:

		print('Loading network weights...')
		net.load(model_data_path, sess)      
		
		print('Initializing network weights...')
		uninitialized_vars = []
		for var in tf.global_variables():
			try:
				sess.run(var)
			except tf.errors.FailedPreconditionError:               
				uninitialized_vars.append(var)

		init_new_vars_op = tf.variables_initializer(uninitialized_vars)
		sess.run(init_new_vars_op)
		
		print('Running data through neural network...')
		pred = sess.run(net.get_output(), feed_dict={input_node: img})
		
		print('Plotting result...')
		fig = plt.figure()
		ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
		fig.colorbar(ii)
		plt.show()

		print('Done...!')
		return pred
		
				
def main():
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('model_path', help='Converted parameters for the model')
	parser.add_argument('image_paths', help='Directory of images to predict')
	args = parser.parse_args()

	# Predict the image
	pred = predict(args.model_path, args.image_paths)
	
	os._exit(0)

if __name__ == '__main__':
	main()
