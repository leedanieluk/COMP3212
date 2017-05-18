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


compare_results('num_results.txt','num_target.txt')