from nltk.translate import AlignedSent
from nltk.translate.ibm1 import IBMModel1
from nltk.tokenize import word_tokenize
import sys

def build_aligned_corpus(num_sents=2000):
	french_sents_raw = []
	eng_sents_raw = []
	french_sents = []
	eng_sents = []

	i = 0
	with open('corpora/fr-en/europarl-v7.fr-en.tok.fr') as fr_file:
		while i < num_sents:
			french_sents_raw.append(fr_file.readline().rstrip())
			i += 1
			
	for sentence in french_sents_raw:
		tokenized_sent = word_tokenize(sentence)
		french_sents.append(tokenized_sent)
	
	i = 0
	with open('corpora/fr-en/europarl-v7.fr-en.tok.en') as en_file:
		while i < num_sents:
			eng_sents_raw.append(en_file.readline().rstrip())
			i += 1
			
	for sentence in eng_sents_raw:
		tokenized_sent = word_tokenize(sentence)
		eng_sents.append(tokenized_sent)
	
	aligned_text = []
	for i in range(0, num_sents):
		algn_snt = AlignedSent(french_sents[i], eng_sents[i])
		aligned_text.append(algn_snt)
	return aligned_text
	
def align_words(bitext):
	ibm1 = IBMModel1(bitext, 5)
	with open('alignment_table.naacl', 'w') as alignment_file:
		for idx, sent_pair in enumerate(bitext):
			alignment_sorted = sorted(sent_pair.alignment)
			for word_pair in alignment_sorted:
				alignment_file.write(str(idx + 1))
				alignment_file.write(' ')
				alignment_file.write(str(word_pair[0]))
				alignment_file.write(' ')
				alignment_file.write(str(word_pair[1]))
				alignment_file.write('\n')

if __name__ == "__main__":
	try:
		aligned_sents = build_aligned_corpus(int(sys.argv[1]))
	except(IndexError):
		aligned_sents = build_aligned_corpus()
	align_words(aligned_sents)
	# print('Building Model...')
	# test_sent = aligned_sents[35]
	# print('Test sentence (English): ', test_sent.words)
	# print('Test sentence (French): ', test_sent.mots)
	# print(test_sent.alignment)
	# fr_word = input('Enter French word: ')
	# en_word = input('Enter English word: ')
	# print('Translation table result: ')
	# print(ibm1.translation_table[fr_word][en_word])