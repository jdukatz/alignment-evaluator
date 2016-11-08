from nltk.translate import AlignedSent, Alignment
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.metrics import alignment_error_rate
import sys

def build_aligned_corpus():
	swed_sents_raw = []
	eng_sents_raw = []
	swed_sents = []
	eng_sents = []

	with open('corpora/sv-en/dev/dev.sv.stripped.naacl') as sv_file:
		for line in sv_file:
			swed_sents_raw.append(line.rstrip())
			
	for sentence in swed_sents_raw:
		tokenized_sent = sentence.split(' ')
		swed_sents.append(tokenized_sent)
	
	with open('corpora/sv-en/dev/dev.en.stripped.naacl') as en_file:
		for line in en_file:
			eng_sents_raw.append(line.rstrip())
			
	for sentence in eng_sents_raw:
		tokenized_sent = sentence.split(' ')
		eng_sents.append(tokenized_sent)
	
	aligned_text = []
	for i in range(0, len(eng_sents)):
		algn_snt = AlignedSent(swed_sents[i], eng_sents[i])
		aligned_text.append(algn_snt)
	return aligned_text
	
def align_words(bitext):
	ibm1 = IBMModel1(bitext, 20)
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
	return bitext
				
def build_gold_corpus():
	"""Read aligned sentences from test set
	read alignment from test sent, then put them
	into an AlignedSent object"""
	#get aligned sentences
	gold_en_sents_raw = []
	with open('corpora/sv-en/dev/dev.en.stripped.naacl') as gold_en_file:
		for line in gold_en_file:
			gold_en_sents_raw.append(line.rstrip())
	
	gold_en_sents = []
	for sentence in gold_en_sents_raw:
		tokenized_sent = sentence.split(' ')
		gold_en_sents.append(tokenized_sent)
	
	gold_sv_sents_raw = []
	with open('corpora/sv-en/dev/dev.sv.stripped.naacl') as gold_sv_file:
		for line in gold_sv_file:
			gold_sv_sents_raw.append(line.rstrip())
	
	gold_sv_sents = []
	for sentence in gold_sv_sents_raw:
		tokenized_sent = sentence.split(' ')
		gold_sv_sents.append(tokenized_sent)

	#get alignments
	alignments = []
	with open('corpora/sv-en/dev/dev.ensv.naacl') as gold_alignment_file:
		aligned_keys_list = []
		idx = 1 #initialize index at one so we don't get a blank entry on the first loop
		prev_idx = 1
		for line in gold_alignment_file:
			alignment_numbers = line.rstrip().split(' ')
			idx = int(alignment_numbers[0])
			if idx != prev_idx: #we've moved to the next sentence so add it to the alignment dictionary
				alignments.append(Alignment(aligned_keys_list))
				aligned_keys_list = []
			aligned_keys = (int(alignment_numbers[1]), int(alignment_numbers[2]))
			aligned_keys_list.append(aligned_keys)
			prev_idx = idx
		alignments.append(Alignment(aligned_keys_list)) #add the last sentence's alignments because the loop above will not drop into the if statement
			
	#put in AlignedSent object
	gold_aligned_sents = []
	for i in range(0, len(gold_en_sents)):
		gold_aligned_sent = AlignedSent(gold_en_sents[i], gold_sv_sents[i], alignments[i])
		gold_aligned_sents.append(gold_aligned_sent)

	return gold_aligned_sents
	
def get_error_rate(alignment, gold_alignment):
	total_error_rate = 0
	for i in range(0, len(alignment)):
		test = alignment[i].alignment
		gold = gold_alignment[i].alignment
		error_rate = alignment_error_rate(gold, test)
		total_error_rate += error_rate
	avg_error_rate = total_error_rate / len(alignment)
	print("average error rate over ", len(alignment), "sentences: ")
	print(avg_error_rate)
	
if __name__ == "__main__":
	bilingual_text = build_aligned_corpus()
	aligned_text = align_words(bilingual_text)
	gold_aligned_text = build_gold_corpus()
	get_error_rate(aligned_text, gold_aligned_text)