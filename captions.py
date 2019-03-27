import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt

import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def read_file(filename):

	# opening file and reading
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text


filename = ".../Flickr8k.token.txt"
# load captions
doc = read_file(filename)
print(doc[:300])

def read_captions(doc):
	mapping = dict()
	# processing data
	for line in doc.split('\n'):
		
		tokens = line.split()
		if len(line) < 2:
			continue
		
		image_id, image_desc = tokens[0], tokens[1:]
		
		image_id = image_id.split('.')[0]
		
		image_desc = ' '.join(image_desc)
		
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping


captions = read_captions(doc)
print('Loaded: %d ' % len(captions))

list(captions.keys())[:5]

captions['3113769557_9edbb8275c']
print('/n')

captions['1000268201_693b08cb0e']

def process_captions(captions):
	
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in captions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
		
			desc = desc.split()
			
			desc = [word.lower() for word in desc]
			
			desc = [w.translate(table) for w in desc]
			
			desc = [word for word in desc if len(word)>1]
			
			desc = [word for word in desc if word.isalpha()]
			
			desc_list[i] =  ' '.join(desc)

process_captions(captions)

def to_vocabulary(captions):
	# build a list of all description strings
	all_desc = set()
	for key in captions.keys():
		[all_desc.update(d.split()) for d in captions[key]]
	return all_desc