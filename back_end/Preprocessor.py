from gensim.corpora import Dictionary
from tqdm import tqdm
import pandas
import Util
import numpy as np
from Configuration import *

class Preprocessor(object):
	def __init__(self):
		self.text_dict = Dictionary()
		self.label_dict = Dictionary()

	def SentencesToVectors(self, splited_sentences):
		vec_sentences = []
		for splited_sentence in tqdm(splited_sentences):
			vec_sentences.append([self.text_dict.token2id.get(word, 0) for word in splited_sentence])
		return vec_sentences

	def LabelsToVectors(self, splited_labels):
		vec_labels = [self.label_dict.token2id[label] for label in splited_labels]
		return vec_labels

	def SaveDicts(self):
		self.text_dict.save(CURRENT_MAIN_PATH + "/dicts/text_dict.dict")
		self.label_dict.save(CURRENT_MAIN_PATH + "/dicts/label_dict.dict")

	def LoadDicts(self):
		self.text_dict = self.text_dict.load(CURRENT_MAIN_PATH + '/dicts/text_dict.dict')
		self.label_dict = self.label_dict.load(CURRENT_MAIN_PATH + '/dicts/label_dict.dict')

	def SaveTrainingData(self, vec_sentences, vec_labels):
		np.savez(CURRENT_MAIN_PATH + "/npz_data/train.npz", x_data = vec_sentences, y_data = vec_labels)

	def SaveTestingData(self, vec_sentences):
		np.savez(CURRENT_MAIN_PATH + "/npz_data/test.npz", x_data = vec_sentences)

	def PreprocessTSV(self, mode = 'train'):
		if mode == 'train':
			filepath = CURRENT_MAIN_PATH + '/raw_data/train.tsv'
			raw_data = pandas.read_csv(filepath, sep = '\t', engine = 'c')
			raw_sentences = raw_data.iloc[:,0]
			raw_triples = raw_data.iloc[:,1]
			# Split by jieba
			print("Spliting sentences and labels")
			splited_sentences = [Util.GetPreProcessedSentence(raw_sentence) for raw_sentence in tqdm(raw_sentences)]
			splited_labels = [Util.GetPreProcessedLabels(raw_triple) for raw_triple in tqdm(raw_triples)]
			# Add words to text dictionary and triple-labels dictionary
			self.text_dict.add_documents(splited_sentences)
			self.label_dict.add_documents([[label] for label in splited_labels])
			# Transform sentences to vectors
			vec_sentences = self.SentencesToVectors(splited_sentences)
			# Transform labels to vectors
			vec_labels = self.LabelsToVectors(splited_labels)
			# Save Dictionaries and training data
			self.SaveDicts()
			self.SaveTrainingData(vec_sentences, vec_labels)
		elif mode == 'test':
			filepath = CURRENT_MAIN_PATH + '/raw_data/test.tsv'
			raw_data = pandas.read_csv(filepath, sep = '\t', engine = 'c')
			raw_sentences = raw_data.iloc[:,0]
			splited_sentences = [Util.GetPreProcessedSentence(raw_sentence) for raw_sentence in tqdm(raw_sentences)]
			# Load Dictionaries
			self.LoadDicts()
			# Transform sentences to vectors
			vec_sentences = self.SentencesToVectors(splited_sentences)
			# Save testing data
			self.SaveTestingData(vec_sentences)

if __name__ == '__main__':
	processor = Preprocessor()
	processor.PreprocessTSV('test')
	# processor.PreprocessTSV('train')