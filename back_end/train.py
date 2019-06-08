from Configuration import *
from VDCNN import *

import numpy as np

from gensim.corpora import Dictionary

import os

import keras
from k_max_pooling import *
from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

label_dict = Dictionary()
label_dict = label_dict.load(CURRENT_MAIN_PATH + '/dicts/label_dict.dict')
total_label = len(label_dict)

print("Total classes : %d" % total_label)

data = np.load(CURRENT_MAIN_PATH + '/npz_data/train.npz')
x_data = data['x_data']
y_data = data['y_data']

# Shuffle the data
indices = np.random.permutation(x_data.shape[0])
x_data = x_data[indices]
y_data = y_data[indices]

x_data = sequence.pad_sequences(x_data, maxlen = SEQ_LEN, padding = 'post', truncating = 'post')
y_data = to_categorical(y_data, num_classes = total_label)

_, _, files = list(os.walk(CURRENT_MAIN_PATH + '/checkpoints/'))[0]
if len(files) > 0:
	losses = [float(name[23:-3]) for name in files]
	losses.sort()
	print("Loading Model : " + '%.2f' % losses[0])
	model = load_model('./checkpoints/vdcnn_weights_val_loss_' + '%.2f' % losses[0] + '.h5', custom_objects = {'KMaxPooling': KMaxPooling})
else:
	model = VDCNN(num_classes = total_label, depth = 49, sequence_length = SEQ_LEN, num_words = 1024, embedding_dim = 16, shortcut = True)

model.compile(optimizer = Adam(lr = 0.00001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
callbacks = [ModelCheckpoint(filepath = CURRENT_MAIN_PATH + "/checkpoints/vdcnn_weights_val_loss_{val_loss:.2f}.h5", period = 1, verbose = 1, save_best_only = False), ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1, patience = 2, verbose = 1)]
if TENSORBOARD:
	callbacks.append(TensorBoard(log_dir='./logs'))
model.fit(x = x_data, y = y_data, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS, validation_split = 0.0001, verbose = 1, callbacks = callbacks)