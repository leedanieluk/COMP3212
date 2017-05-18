import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random
import pickle
from collections import Counter
from numpy import matrix
from collections import Iterable

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def get_aminoacid(training):
	for fi in [training]:
		with open(fi,'r') as f:
			contents = f.readlines()
			aminoacid = []
			for l in contents[:19025]:
				all_pairs = word_tokenize(l)
				aminoacid.append(all_pairs[0])
	return aminoacid

def get_struc_type(training):
	for fi in [training]:
		with open(fi,'r') as f:
			contents = f.readlines()
			struct_type = []
			for l in contents[:19025]:
				all_pairs = word_tokenize(l)
				struct_type.append(all_pairs[1])
	return struct_type

def get_aminoacid_test(training):
	for fi in [training]:
		with open(fi,'r') as f:
			contents = f.readlines()
			aminoacid = []
			for l in contents[:3680]:
				all_pairs = word_tokenize(l)
				aminoacid.append(all_pairs[0])
	return aminoacid

def get_struc_type_test(training):
	for fi in [training]:
		with open(fi,'r') as f:
			contents = f.readlines()
			struct_type = []
			for l in contents[:3680]:
				all_pairs = word_tokenize(l)
				struct_type.append(all_pairs[1])
	return struct_type

def get_training_features(training_file, kernel_size):
	featureset = []

	aminoacid = []
	struct_type = []
	aminoacid = get_aminoacid(training_file)
	struct_type = get_struc_type(training_file)

	# get feature and classes for each aminoacid
	for index, aa in enumerate(aminoacid):
		temp_feature = []
		temp_classification = []

		# get feature
		if aa == 'end':
			continue
		else:
			for i in range(-kernel_size,kernel_size+1):
				# check for end
				if aminoacid[index+i] == 'end':
					empty_row = [0]*20
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'A':
					empty_row = [0]*20
					empty_row[0] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'C':
					empty_row = [0]*20
					empty_row[1] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'D':
					empty_row = [0]*20
					empty_row[2] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'E':
					empty_row = [0]*20
					empty_row[3] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'F':
					empty_row = [0]*20
					empty_row[4] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'G':
					empty_row = [0]*20
					empty_row[5] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'H':
					empty_row = [0]*20
					empty_row[6] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'I':
					empty_row = [0]*20
					empty_row[7] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'K':
					empty_row = [0]*20
					empty_row[8] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'L':
					empty_row = [0]*20
					empty_row[9] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'M':
					empty_row = [0]*20
					empty_row[10] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'N':
					empty_row = [0]*20
					empty_row[11] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'P':
					empty_row = [0]*20
					empty_row[12] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'Q':
					empty_row = [0]*20
					empty_row[13] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'R':
					empty_row = [0]*20
					empty_row[14] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'S':
					empty_row = [0]*20
					empty_row[15] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'T':
					empty_row = [0]*20
					empty_row[16] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'V':
					empty_row = [0]*20
					empty_row[17] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'W':
					empty_row = [0]*20
					empty_row[18] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'Y':
					empty_row = [0]*20
					empty_row[19] = 1
					temp_feature.append(empty_row)
		
		# get structure type / classification
		if struct_type[index] == 'x':
			continue
		elif struct_type[index] == '_':
			empty_row = [0]*3
			empty_row[0] = 1
			temp_classification.append(empty_row)

		elif struct_type[index] == 'e':
			empty_row = [0]*3
			empty_row[1] = 1
			temp_classification.append(empty_row)

		elif struct_type[index] == 'h':
			empty_row = [0]*3
			empty_row[2] = 1
			temp_classification.append(empty_row)

		# store it in feature set
		temp_feature = list(temp_feature)
		temp_feature = flatten(temp_feature)

		temp_classification = list(temp_classification)
		temp_classification = flatten(temp_classification)
		
		featureset.append([temp_feature,temp_classification])

	return featureset

def get_testing_features(testing_file, kernel_size):
	featureset = []

	aminoacid = []
	struct_type = []
	aminoacid = get_aminoacid_test(testing_file)
	struct_type = get_struc_type_test(testing_file)

	# get feature and classes for each aminoacid
	for index, aa in enumerate(aminoacid):
		temp_feature = []
		temp_classification = []

		# get feature
		if aa == 'end':
			continue
		else:
			for i in range(-kernel_size,kernel_size+1):
				# check for end
				if aminoacid[index+i] == 'end':
					empty_row = [0]*20
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'A':
					empty_row = [0]*20
					empty_row[0] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'C':
					empty_row = [0]*20
					empty_row[1] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'D':
					empty_row = [0]*20
					empty_row[2] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'E':
					empty_row = [0]*20
					empty_row[3] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'F':
					empty_row = [0]*20
					empty_row[4] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'G':
					empty_row = [0]*20
					empty_row[5] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'H':
					empty_row = [0]*20
					empty_row[6] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'I':
					empty_row = [0]*20
					empty_row[7] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'K':
					empty_row = [0]*20
					empty_row[8] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'L':
					empty_row = [0]*20
					empty_row[9] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'M':
					empty_row = [0]*20
					empty_row[10] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'N':
					empty_row = [0]*20
					empty_row[11] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'P':
					empty_row = [0]*20
					empty_row[12] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'Q':
					empty_row = [0]*20
					empty_row[13] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'R':
					empty_row = [0]*20
					empty_row[14] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'S':
					empty_row = [0]*20
					empty_row[15] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'T':
					empty_row = [0]*20
					empty_row[16] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'V':
					empty_row = [0]*20
					empty_row[17] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'W':
					empty_row = [0]*20
					empty_row[18] = 1
					temp_feature.append(empty_row)

				elif aminoacid[index+i] == 'Y':
					empty_row = [0]*20
					empty_row[19] = 1
					temp_feature.append(empty_row)
		
		# get structure type / classification
		if struct_type[index] == 'x':
			continue
		elif struct_type[index] == '_':
			empty_row = [0]*3
			empty_row[0] = 1
			temp_classification.append(empty_row)

		elif struct_type[index] == 'e':
			empty_row = [0]*3
			empty_row[1] = 1
			temp_classification.append(empty_row)

		elif struct_type[index] == 'h':
			empty_row = [0]*3
			empty_row[2] = 1
			temp_classification.append(empty_row)

		# store it in feature set
		temp_feature = list(temp_feature)
		temp_feature = flatten(temp_feature)

		temp_classification = list(temp_classification)
		temp_classification = flatten(temp_classification)

		featureset.append([temp_feature,temp_classification])

	return featureset


def create_feature_sets_and_labels(kernel_size):
	training_features = []
	testing_features = []

	training_features = get_training_features('training_set.txt', kernel_size)
	testing_features = get_testing_features('testing_set.txt', kernel_size)
	random.shuffle(training_features)

	training_features = np.array(training_features)
	testing_features = np.array(testing_features)

	train_x = list(training_features[:,0])
	train_y = list(training_features[:,1])

	test_x = list(testing_features[:,0].flatten())
	test_y = list(testing_features[:,1])

	return train_x,train_y,test_x,test_y

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels(10)
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
















