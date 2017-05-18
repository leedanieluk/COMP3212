import nltk
import ast
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random
import pickle
from collections import Counter
from numpy import matrix
from collections import Iterable

def get_targets(target_file):

	targets = []

	for fi in [target_file]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:len(contents)]:
				targets.append(word_tokenize(l))

	return targets


def get_results(results_file):

	results = []

	for fi in [results_file]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:len(contents)]:
				results.append(word_tokenize(l))

	return results

def get_accuracy(target_file, results_file):


	targets = get_targets(target_file)
	results = get_results(results_file)

	structure = []
	structure.append(word_tokenize('h'))
	structure.append(word_tokenize('e'))
	structure.append(word_tokenize('_'))

	matches = 0

	helix_count = 0
	helix_match = 0
	helix_mismatch = 0

	sheet_count = 0
	sheet_match = 0
	sheet_mismatch = 0

	coil_count = 0
	coil_match = 0
	coil_mismatch = 0

	total_count = 0

	for index, target in enumerate(targets):
		total_count += 1

		if target == structure[0]:
			helix_count += 1

		if target == structure[1]:
			sheet_count += 1

		if target == structure[2]:
			coil_count += 1


		if results[index] == target:
			matches += 1
			if target == structure[0]:
				helix_match += 1
			elif target == structure[1]:
				sheet_match += 1
			elif target == structure[2]:
				coil_match += 1
		else: 
			continue

	accuracy = matches / len(targets)

	print(results_file,': ',accuracy)

	if helix_count > 0:
		helix_accuracy = helix_match / helix_count
		print('Helix Accuracy: ', helix_accuracy)
	else:
		print('No Helix aminoacid')

	if sheet_count > 0:
		sheet_accuracy = sheet_match / sheet_count
		print('Sheet Accuracy: ', sheet_accuracy)
	else:
		print('No Sheet aminoacid')

	if coil_count > 0:
		coil_accuracy = coil_match / coil_count
		print('Coil Accuracy: ', coil_accuracy)
	else:
		print('No Coil aminoacid')

	print('Length: ', total_count)

	return accuracy

get_accuracy('./targets/seq1_target.txt','./results/seq1_results.txt')
get_accuracy('./targets/seq2_target.txt','./results/seq2_results.txt')
get_accuracy('./targets/seq3_target.txt','./results/seq3_results.txt')
get_accuracy('./targets/seq4_target.txt','./results/seq4_results.txt')
get_accuracy('./targets/seq5_target.txt','./results/seq5_results.txt')
get_accuracy('./targets/seq6_target.txt','./results/seq6_results.txt')
get_accuracy('./targets/seq7_target.txt','./results/seq7_results.txt')
get_accuracy('./targets/seq8_target.txt','./results/seq8_results.txt')
get_accuracy('./targets/seq9_target.txt','./results/seq9_results.txt')
get_accuracy('./targets/seq10_target.txt','./results/seq10_results.txt')
get_accuracy('./targets/seq11_target.txt','./results/seq11_results.txt')
get_accuracy('./targets/seq12_target.txt','./results/seq12_results.txt')
get_accuracy('./targets/seq13_target.txt','./results/seq13_results.txt')
get_accuracy('./targets/seq14_target.txt','./results/seq14_results.txt')
get_accuracy('./targets/seq15_target.txt','./results/seq15_results.txt')

















