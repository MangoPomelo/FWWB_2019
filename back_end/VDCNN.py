import keras
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, Dense, Flatten
from k_max_pooling import *
from keras.utils import plot_model

def Convolutional_Block(inputs, filters, kernel_size = 3, use_bias = False, shortcut = False, pool_type = 'max', sorted = True, downsample = False, stage = 1):
	conv1 = Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same')(inputs)
	bn1 = BatchNormalization()(conv1)
	relu1 = Activation('relu')(bn1)

	conv2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same')(relu1)
	relu2 = BatchNormalization()(conv2)
	out = Activation('relu')(relu2)

	if downsample:
		out = Downsample(out, pool_type = pool_type, sorted = sorted, stage = stage)

	if shortcut:
		residual = Conv1D(filters = filters, kernel_size = 1, strides = 2 if downsample else 1)(inputs)
		residual = BatchNormalization()(residual)
		out = Add()([out, residual])

	return out

def Downsample(inputs, pool_type = 'max', sorted = True, stage = 1):
	if pool_type == 'max':
		out = MaxPooling1D(pool_size = 3, strides = 2, padding = 'same', name= 'pool_%d' % stage)(inputs)
	elif pool_type == 'k_max':
		k = int(inputs._keras_shape[1]/2)
		out = KMaxPooling(k = k, sorted = sorted, name='pool_%d' % stage)(inputs)
	elif pool_type is None:
		out = inputs
	else:
		raise ValueError('unsupported pooling type!')
	return out

def VDCNN(num_classes, depth = 9, sequence_length = 256, num_words = 1024, embedding_dim = 16, shortcut = False, pool_type = "max", sorted = True, use_bias = False, input_tensor = None):
	if depth == 9:
		num_conv_blocks = (1, 1, 1, 1)
	elif depth == 17:
		num_conv_blocks = (2, 2, 2, 2)
	elif depth == 29:
		num_conv_blocks = (2, 2, 5, 5)
	elif depth == 49:
		num_conv_blocks = (8, 8, 5, 3)
	else:
		raise ValueError('unsupported depth for VDCNN.')

	inputs = Input(shape = (sequence_length,), name = 'inputs')
	embedded_words = Embedding(input_dim = num_words, output_dim = embedding_dim)(inputs)
	out = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name='temp_conv')(embedded_words)

	for i in range(num_conv_blocks[0] - 1):
		out = Convolutional_Block(out, filters = 64, kernel_size = 3, use_bias = use_bias, shortcut = shortcut, pool_type = None, sorted = sorted, downsample = False, stage = 1)
	out = Convolutional_Block(out, filters = 64, kernel_size = 3, use_bias = use_bias, shortcut = shortcut, pool_type = pool_type, sorted = sorted, downsample = True, stage = 1)

	for i in range(num_conv_blocks[1] - 1):
		out = Convolutional_Block(out, filters = 128, kernel_size = 3, use_bias = use_bias, shortcut = shortcut, pool_type = None, sorted = sorted, downsample = False, stage = 2)
	out = Convolutional_Block(out, filters = 128, kernel_size = 3, use_bias = use_bias, shortcut = shortcut, pool_type = pool_type, sorted = sorted, downsample = True, stage = 2)

	for i in range(num_conv_blocks[2] - 1):
		out = Convolutional_Block(out, filters = 256, kernel_size = 3, use_bias = use_bias, shortcut = shortcut, pool_type = None, sorted = sorted, downsample = False, stage = 3)
	out = Convolutional_Block(out, filters = 256, kernel_size = 3, use_bias = use_bias, shortcut = shortcut, pool_type = pool_type, sorted = sorted, downsample = True, stage = 3)

	for i in range(num_conv_blocks[3] - 1):
		out = Convolutional_Block(out, filters = 512, kernel_size = 3, use_bias = use_bias, shortcut = shortcut, pool_type = None, sorted = sorted, downsample = False, stage = 4)
	out = Convolutional_Block(out, filters = 512, kernel_size = 3, use_bias = use_bias, shortcut = shortcut, pool_type = pool_type, sorted = sorted, downsample = True, stage = 4)
	
	out = KMaxPooling(k = 8, sorted = True)(out)	
	out = Flatten()(out)

	
	out = Dense(2048, activation = 'relu')(out)
	out = Dense(2048, activation = 'relu')(out)
	out = Dense(num_classes, activation = 'softmax', name = 'out')(out)
	
	model = Model(inputs = inputs, outputs = out, name = 'VDCNN')
	return model

if __name__ == "__main__":
	model = VDCNN(1024, depth = 17, shortcut = True, pool_type = 'max')
	model.summary()
	plot_model(model, to_file = "./arch/VDCNN.png", show_shapes = True)