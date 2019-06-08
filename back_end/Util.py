import jieba
import re

re_symbols = re.compile(r'[-,$()#+&*\d:/（）【】.．]')

def GetPreProcessedSentence(raw_sentence):
	raw_sentence = jieba.lcut(re.sub(re_symbols, '', raw_sentence), HMM = True)
	while ' ' in raw_sentence:
		raw_sentence.remove(' ')
	return raw_sentence

def GetPreProcessedLabels(raw_labels):
	return raw_labels