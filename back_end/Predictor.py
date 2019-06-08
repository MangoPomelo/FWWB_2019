import Util
from gensim.corpora import Dictionary
from keras.models import load_model
from Configuration import *
from k_max_pooling import *
from keras.preprocessing import sequence

class Predictor:
	def __init__(self):
		self.model = load_model('./checkpoints/vdcnn_weights_best.h5', custom_objects = {'KMaxPooling': KMaxPooling})
		# Load dicts
		self.text_dict, self.label_dict = Dictionary(), Dictionary()
		self.text_dict = self.text_dict.load(CURRENT_MAIN_PATH + '/dicts/text_dict.dict')
		self.label_dict = self.label_dict.load(CURRENT_MAIN_PATH + '/dicts/label_dict.dict')
	def predict(self, raw_sentence):
		# raw 2 vec
		splited_sentence = Util.GetPreProcessedSentence(raw_sentence)
		vec_sentence = [self.text_dict.token2id.get(word, 0) for word in splited_sentence]
		vec_sentence = sequence.pad_sequences([vec_sentence], maxlen = SEQ_LEN, padding = 'post', truncating = 'post')
		# vec 4 predict
		vec_labels = self.model.predict(vec_sentence)
		vec_labels = [np.argmax(vec_label) for vec_label in vec_labels][0]
		# vec_result 2 nl
		return self.label_dict[vec_labels]

if __name__ == '__main__':
	pd = Predictor()
	print(pd.predict("123"))
