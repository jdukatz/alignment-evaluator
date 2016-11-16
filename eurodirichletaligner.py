from nltk.translate import AlignedSent, Alignment
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.metrics import alignment_error_rate
from collections import defaultdict
from np.random import gamma, dirichlet

def build_aligned_corpus():
	print('building training corpus...')
	de_sents_raw = []
	eng_sents_raw = []
	de_sents = []
	eng_sents = []

	with open('corpora/DeEn/europarl-v7.de-en.tok.de', encoding='cp437') as de_file:
	#USE HEAD FILE FOR TESTING:
	#with open('corpora/DeEn/europarl-v7.de-en.head.tok.de') as de_file:
		for line in de_file:
			de_sents_raw.append(line.rstrip())
			
	for sentence in de_sents_raw:
		tokenized_sent = sentence.split(' ')
		de_sents.append(tokenized_sent)
	
	with open('corpora/DeEn/europarl-v7.de-en.tok.en', encoding='cp437') as en_file:
	#USE HEAD FILE FOR TESTING
	#with open('corpora/DeEn/europarl-v7.de-en.head.tok.en') as en_file:
		for line in en_file:
			eng_sents_raw.append(line.rstrip())
			
	for sentence in eng_sents_raw:
		tokenized_sent = sentence.split(' ')
		eng_sents.append(tokenized_sent)
	
	aligned_text = []
	for i in range(0, len(eng_sents)):
		algn_snt = AlignedSent(eng_sents[i], de_sents[i])
		aligned_text.append(algn_snt)
		
	return aligned_text
	
def align_words(bitext):
	print('building alignment model..')
		"""
	Feeds prior probabilities drawn from a Dirchlet distribution
	into IBM model 1
	Dirichlet distribution is set up as a grid of source and target words, eg:
	
				trg_word_1	trg_word_2	trg_word_3	...
	src_word_1	value		value		value		...
	src_word_2	value		value		value		...
	src_word_3	value		value		value		...
	...			...			...			...
	"""
	
	src_vocab = set()
	trg_vocab = set()
	for aligned_sentence in bitext:
		src_vocab.update(aligned_sentence.words)
		trg_vocab.update(aligned_sentence.mots)
	
	#fill in with the same default value as IBMModel
	#we'll overwrite this in a moment
	dist = defaultdict(lambda: defaultdict(lambda: 1.0e-12))
	
	n = len(src_vocab)
	#potential options for seeding dirichlet distribution; still looking for best solution
	probs = gamma(5, size=len(trg_vocab))
	#probs = np.random.randint(1, high=9, size=len(trg_vocab))
	alphas = [x for x in probs]
	dirichlet_probs = dirichlet(alphas, n)
	
	for src_idx, src_word in enumerate(src_vocab):
		for trg_idx, trg_word in enumerate(trg_vocab):
			dist[src_word][trg_word] = dirichlet_probs[src_idx][trg_idx]

	probability_tables = {}
	probability_tables['translation_table'] = dist
	ibm1 = IBMModel1(bitext, 1, probability_tables)
	print('Model complete')
	print('Aligning eval corpus...')
	
	# with open('basic_alignment_table.naacl', 'w') as alignment_file:
		# for idx, sent_pair in enumerate(bitext):
			# alignment_sorted = sorted(sent_pair.alignment)
			# for word_pair in alignment_sorted:
				# alignment_file.write(str(idx + 1))
				# alignment_file.write(' ')
				# alignment_file.write(str(word_pair[0]))
				# alignment_file.write(' ')
				# alignment_file.write(str(word_pair[1]))
				# alignment_file.write('\n')
				
	#build corpus to be test aligned
	de_sents_raw = []
	eng_sents_raw = []
	de_sents = []
	eng_sents = []
	
	with open('corpora/DeEn/gold-aligned-corpus/de') as de_file:
		for line in de_file:
			de_sents_raw.append(line.rstrip())
			
	for sentence in de_sents_raw:
		tokenized_sent = sentence.split(' ')
		de_sents.append(tokenized_sent)
	
	with open('corpora/DeEn/gold-aligned-corpus/en') as en_file:
		for line in en_file:
			eng_sents_raw.append(line.rstrip())
			
	for sentence in eng_sents_raw:
		tokenized_sent = sentence.split(' ')
		eng_sents.append(tokenized_sent)
		
	aligned_test_text = []
	for i in range(0, len(eng_sents)):
		algn_test_snt = AlignedSent(de_sents[i], eng_sents[i])
		aligned_test_text.append(algn_test_snt)
		
	#use trained model to run test alignment
	ibm1._IBMModel1__align_all(aligned_test_text)
	
	return aligned_test_text

def build_gold_corpus():
	"""Read aligned sentences from test set
	read alignment from test sent, then put them
	into an AlignedSent object"""
	print('building gold corpus...')
	#get aligned sentences
	gold_en_sents_raw = []
	with open('corpora/DeEn/gold-aligned-corpus/en') as gold_en_file:
		for line in gold_en_file:
			gold_en_sents_raw.append(line.rstrip())
	
	gold_en_sents = []
	for sentence in gold_en_sents_raw:
		tokenized_sent = sentence.split(' ')
		gold_en_sents.append(tokenized_sent)
	
	gold_de_sents_raw = []
	with open('corpora/DeEn/gold-aligned-corpus/de') as gold_de_file:
		for line in gold_de_file:
			gold_de_sents_raw.append(line.rstrip())
	
	gold_de_sents = []
	for sentence in gold_de_sents_raw:
		tokenized_sent = sentence.split(' ')
		gold_de_sents.append(tokenized_sent)

	#get alignments
	alignments = [] #a list of Alignment objects, one for each sentence
	aligned_keys_list = [] #a list of tuples, one for each word alignment in a sentence; will be cleared each sentence
	with open('corpora/DeEn/gold-aligned-corpus/alignmentDeEn') as gold_alignment_file:
		for line in gold_alignment_file:
			if line[:4] == 'SENT':
				continue
			elif line == '\n':
				#blank line means new sentence
				#so put the current sentence in an Alignment object,
				#add it to the list, and empty the contents of the list
				alignments.append(Alignment(aligned_keys_list))
				del aligned_keys_list[:]
			else:
				alignment_numbers = line.rstrip().split(' ')
				aligned_keys = (int(alignment_numbers[1]), int(alignment_numbers[2]))
				aligned_keys_list.append(aligned_keys)
			
	#put in AlignedSent object
	gold_aligned_sents = []
	for i in range(0, len(gold_en_sents) - 1):
		# print(i)
		# print(gold_de_sents[i])
		# print(gold_en_sents[i])
		# print(alignments[i])
		gold_aligned_sent = AlignedSent(gold_de_sents[i], gold_en_sents[i], alignments[i])
		gold_aligned_sents.append(gold_aligned_sent)

	return gold_aligned_sents
	
def get_error_rate(alignment, gold_alignment):
	print('calculating error rate...')
	total_error_rate = 0
	for i in range(0, len(alignment) - 1):
		test = alignment[i].alignment
		gold = gold_alignment[i].alignment
		error_rate = alignment_error_rate(gold, test)
		total_error_rate += error_rate
	avg_error_rate = total_error_rate / len(alignment)
	print("average error rate over ", len(alignment), "sentences: ")
	print(avg_error_rate)
	
if __name__ == "__main__":
	bilingual_text = build_aligned_corpus()
	test_aligned_text = align_words(bilingual_text)
	# for sentence in test_aligned_text:
		# print(sentence.alignment)
	gold_aligned_text = build_gold_corpus()
	get_error_rate(test_aligned_text, gold_aligned_text)