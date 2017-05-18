import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import nltk
import ast
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from read_file import create_feature_sets_and_labels
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# check the data structure
# mnist = input_data.read_data_sets("/tmp/data", one_hot=True) # one_hot = output([1 0 0])

train_x,train_y,test_x,test_y = create_feature_sets_and_labels(10)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 3
batch_size = 20

# x' = pixels of the image (width, height)
x = tf.placeholder('float',[None, len(train_x[0])])
y = tf.placeholder('float')

#Function with dictionaries[matrix]
def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}	  

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}	  

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}	  


	# (input_data * weights) + biases = input to hidden layers (before activation function)
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
	#output = tf.nn.relu(output)

	return output

def train_neural_network(x):
	# 1. get prediction, calculate cost
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

	# stochastic gradient descent
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# iterations
	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0

			i = 0
			while i < len(train_x):
				start = i
				end = i+batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c

				i += batch_size

			print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss', epoch_loss)

		# check which index is highest and compare with answer
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		output = sess.run(tf.argmax(prediction,1),feed_dict={x: test_x})
		
		for index, value in enumerate(output):
			print(value)

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


def compare_results(results_file,targets_file):

	results = []
	targets = []

	structure_type = []
	structure_type.append(word_tokenize('_')) # _
	structure_type.append(word_tokenize('e')) # e	
	structure_type.append(word_tokenize('h')) # h

	coil_match = 0
	sheet_match = 0
	helix_match = 0

	coil_count = 0
	sheet_count = 0
	helix_count = 0


	for fi in [results_file]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:len(contents)]:
				results.append(word_tokenize(l))

	for fi in [targets_file]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:len(contents)]:
				targets.append(word_tokenize(l))

	for index, value in enumerate(targets):

		if value == structure_type[0]:
			coil_count += 1

		if value == structure_type[1]:
			sheet_count += 1

		if value == structure_type[2]:
			helix_count += 1

		if results[index] == value:
			if value == structure_type[0]:
				coil_match += 1
			elif value == structure_type[1]:
				sheet_match += 1
			elif value == structure_type[2]:
				helix_match += 1

	print('Coil:',coil_match/coil_count)
	print('Sheet:',sheet_match/sheet_count)
	print('Helix:',helix_match/helix_count)

	return results

train_neural_network(x)
compare_results('num_results.txt','num_target.txt')














