from Configuration import *

from gensim.corpora import Dictionary
from k_max_pooling import *
from keras.models import load_model
from keras.preprocessing import sequence
from tqdm import tqdm


import os
import numpy as np
import itertools

# Load dicts
label_dict = Dictionary()
label_dict = label_dict.load(CURRENT_MAIN_PATH + '/dicts/label_dict.dict')

# Load data
data = np.load(CURRENT_MAIN_PATH + '/npz_data/test.npz')
partial_x_datas = np.array_split(data['x_data'], DIVISION_NUM, axis = 0)

# Load Model which has min loss
_, _, files = list(os.walk(CURRENT_MAIN_PATH + '/checkpoints/'))[0]
losses = [float(name[23:-3]) for name in files]
losses.sort()
print("Loading Model : " + '%.2f' % losses[0])
model = load_model('./checkpoints/vdcnn_weights_val_loss_' + '%.2f' % losses[0] + '.h5', custom_objects = {'KMaxPooling': KMaxPooling})

with open(CURRENT_MAIN_PATH + '/raw_data/test.tsv', 'r', encoding = 'utf-8') as test_file:
	for part_num, partial_x_data in enumerate(partial_x_datas):
		part_num += 1
		print("Predicting the part %d of %d" % (part_num, DIVISION_NUM))
		partial_x_data = sequence.pad_sequences(tqdm(partial_x_data), maxlen = SEQ_LEN, padding = 'post', truncating = 'post')
		vec_labels = (np.argmax(vec_label) for vec_label in model.predict(partial_x_data, verbose = 1))
		labels = itertools.chain([""], (label_dict[label] for label in vec_labels)) if part_num == 1 else (label_dict[label] for label in vec_labels)

		# Write into file
		with open(CURRENT_MAIN_PATH + '/result/result.part%d.tsv' % part_num, 'w', encoding = 'utf-8') as result_file:
				content = []
				print("Writing into file ...")
				for label in tqdm(labels):
					content.append(next(test_file).strip('\n') + '\t' + label + '\n')
				result_file.writelines(content)
				print('Result has been successfully saved at "/result/result.part%d.tsv"' % part_num)