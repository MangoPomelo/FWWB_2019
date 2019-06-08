from keras.engine import Layer, InputSpec
from keras.layers import Flatten
import tensorflow as tf
import numpy as np

class KMaxPooling(Layer):
	def __init__(self, k = 1, sorted = True, **kwargs):
		super().__init__(**kwargs)
		self.input_spec = InputSpec(ndim = 3)
		self.k = k
		self.sorted = sorted

	def compute_out_shape(self, input_shape):
		return input_shape

	def call(self, inputs):
		input_shape = tf.shape(inputs)
		delta = input_shape[1] - self.k
		shifted_inputs = tf.transpose(inputs, [0, 2, 1])
		fill = tf.zeros([input_shape[0], input_shape[2], delta])
		top_k = tf.concat([tf.nn.top_k(shifted_inputs, k = self.k, sorted = self.sorted)[0], fill], axis = 2)
		return tf.transpose(top_k, [0,2,1])