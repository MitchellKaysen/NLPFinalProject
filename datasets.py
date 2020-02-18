import torch.utils.data as data
import re
import os
import torch
from utils import *
import numpy as np
from collections import Counter

def load_glove(emb_dim):
	path = 'glove.6B/glove.6B.' + str(emb_dim) + 'd.txt'
	emb = {}
	with open(path, encoding='UTF-8') as f:
		for line in f:

			l = line.split(' ')
			emb[l[0]] = np.array([float(v) for v in l[1:]])		
	return emb

def _count(counter, s):
	for tok in s:
		counter[tok] += 1
"""
	Dataset structure:
	0 or more lines of persona info:
		your persona: text
	0 or more lines of persona info:
		partner's persona: text
	1 or more lines of dialog turns:
		partner utterance \t self utterance \t candidate 1 | ... | candidate n
		candidate replies used for ranking model so will simply be discarded as
		only considering a generative model for this project.
"""
class PersonaDataset(data.Dataset):
	"""
	"""
	def __init__(self, subset, both, revision, indexer):
		self.ML = {
			'sent': 0,
			'persona' : 0,
			'n_sent' : 0,
			'n_pers' : 0
		}
		max_len_p = 0
		max_len_h = 0
		max_n_sent = 0
		max_n_pers = 0
		self.pad = indexer.add_and_get_index('<pad>')
		SOS = indexer.add_and_get_index('<s>')
		EOS = indexer.add_and_get_index('<eos>') #more end of utterance than end of sentence tbh
		path = os.path.join("personachat/", (subset + "_" + both + "_" + revision + ".txt"))
		assert(os.path.exists(path))
		print(path)
		self.data = []
		self.counts = Counter()
		episode = {
			's_per' : [],
			'p_per' : [],
			's_hist' : [],
			'p_hist' : [],
			'labels' : []
		}
		samples = 0
		with open(path) as file:
			read_first_episode = True
			line = file.readline()
			#read line
			ep = 0
			while line != "":

				line_num, _, text = line.strip().partition(' ')
				if int(line_num) == 1 and not read_first_episode:
					max_n_pers = max(max_n_pers, max(len(episode['p_per']), len(episode['s_per'])))
					max_n_sent = max(max_n_sent, max(len(episode['p_hist']), len(episode['s_hist'])))
					self.data.append(episode)
					episode = {
						's_per' : [],
						'p_per' : [],
						's_hist' : [],
						'labels' : [],
						'p_hist' : []
					}
					ep+= 1

				else:
					read_first_episode = False

				text = text.replace(".", " .").replace("  ", " ")
				#may have to replace "n't" to ' not'
				if re.search(r'^your persona:', text):
					sent = text.replace("your persona: ", '').split(' ')
					idxs = [SOS] + [indexer.add_and_get_index(s) for s in sent] + [EOS]
					_count(self.counts, idxs)
					episode['s_per'].append(idxs)
					max_len_p = max(max_len_p, len(idxs))
				elif re.search(r"^partner's persona:", text):
					sent = text.replace("partner's persona: ", '').split(' ')
					idxs = [SOS] + [indexer.add_and_get_index(s) for s in sent] + [EOS]
					_count(self.counts, idxs)
					episode['p_per'].append(idxs)
					max_len_p = max(max_len_p, len(idxs))
				else:
					sents = text.split('\t')[:2]
					turn = [[SOS] + [indexer.add_and_get_index(s) for s in sent.split(' ')] + [EOS] for sent in sents]
					for sent in turn:
						_count(self.counts, sent)
					episode['s_hist'].append(turn[1][:-1])
					episode['labels'].append(turn[1][1:])
					episode['p_hist'].append(turn[0])

					max_len_h = max(max_len_h, max(len(turn[0]), len(turn[1])))

				line = file.readline()
		self.ML = {
			'sent' : max_len_h,
			'persona': max_len_p,
			'n_pers' : max_n_pers,
			'n_sent' : max_n_sent
		}

	#this is fucking blinkered, the padding only matters for the current batch...
	def __getitem__(self, idx):
		item = self.data[idx]
		max_l = self.ML['sent']
		max_p = self.ML['persona']
		max_n_pers = self.ML['n_pers']

		max_n_sent = self.ML['n_sent']
		pad = self.pad

		persona_pad = [pad for _ in range(max_p)]
		sentence_pad = [pad for _ in range(max_l)]
		x = {}

		#pad each persona and persona histories
		self_pad = []
		
		for s in item['s_per']:
			self_pad.append(s + persona_pad[len(s):])
		part_pad = []
		for s in item['p_per']:
			part_pad.append(s + persona_pad[len(s):])
		for _ in range(max_n_pers - len(item['s_per'])):
			self_pad.append(persona_pad)
		for _ in range(max_n_pers - len(item['p_per'])):
			part_pad.append(persona_pad)

		#pad each sentence and pad conversation histories
		self_history_pad = []
		part_history_pad = []
		labels_pad = []

		l_s_hist = []
		for s in item['s_hist']:
			l_s_hist.append(len(s))
			self_history_pad.append(s + sentence_pad[len(s):])
		for s in item['p_hist']:
			part_history_pad.append(s + sentence_pad[len(s):])
		for s in item['labels']:
			labels_pad.append(s + sentence_pad[len(s):])
		for _ in range(max_n_sent - len(self_history_pad)):
			self_history_pad.append(sentence_pad)
		for _ in range(max_n_sent - len(part_history_pad)):
			part_history_pad.append(sentence_pad)
		for _ in range(max_n_sent - len(labels_pad)):
			labels_pad.append(sentence_pad)
		
		return self_pad, self_history_pad, part_pad, part_history_pad, labels_pad

	def __len__(self):
		return len(self.data)

#not carrying hidden state between dialogue turns
class PersonaDatasetAlt(data.Dataset):
	"""
	"""
	def __init__(self, subset, both, revision):
		max_len_p = 0
		max_len_h = 0

		SOS = '<s>'
		EOS = '<eos>' #indexer.add_and_get_index('<eos>') #more end of utterance than end of sentence tbh
		path = os.path.join("personachat/", (subset + "_" + both + "_" + revision + ".txt"))
		assert(os.path.exists(path))
		print(path)
		self.data = []
		self.personas = []
		self.counts = Counter()
		persona = {
			'self' : [],
			'partner' : []
		}
		samples = 0
		with open(path) as file:
			read_first_episode = True
			line = file.readline()
			#read line
			ep_idx = 0
			while line != "":

				line_num, _, text = line.strip().partition(' ')
				if int(line_num) == 1 and not read_first_episode:
					persona['self'] = [SOS] + persona['self'] + [EOS] #need EOS for cat
					persona['partner'] = [SOS] + persona['partner'] + [EOS]
					_count(self.counts, idxs)
					self.personas.append(persona)
					persona = {
						'self' : [],
						'partner' : []
					}
					ep_idx += 1

				else:
					read_first_episode = False

				text = text.replace(".", " .").replace("  ", " ")
				#may have to replace "n't" to ' not'
				if re.search(r'^your persona:', text):
					sent = text.replace("your persona: ", '').split(' ')
					idxs = sent
					_count(self.counts, idxs)
					persona['self'] += idxs
					max_len_p = max(max_len_p, len(idxs))
				elif re.search(r"^partner's persona:", text):
					sent = text.replace("partner's persona: ", '').split(' ')
					idxs =  sent
					_count(self.counts, idxs)
					persona['partner'] += idxs
					max_len_p = max(max_len_p, len(idxs))
				else:
					sents = text.split('\t')[:2]
					turn = [[SOS] + sent.split(' ') + [EOS] for sent in sents]
					for sent in turn:
						_count(self.counts, sent)
					self.data.append((turn[0], turn[1], ep_idx))
					max_len_h = max(max_len_h, max(len(turn[0]), len(turn[1])))

				line = file.readline()
			self.personas.append(persona)
			
		self.max_len_p = max_len_p
		self.max_len_h = max_len_h

	def __getitem__(self, idx):
		them, me, episode = self.data[idx]
		try:
			self.personas[episode]
		except Exception:
			print(episode)
			print(len(self.personas))
			raise Exception()
		return self.personas[episode], them, me

	def __len__(self):
		return len(self.data)

class BabiData(data.Dataset):

	def  __init__(self, task, train, indexer):
		self.max_len = 0
		path = task + '_' + train + '.txt'
		EOS = indexer.add_and_get_index('<EOS>')
		assert(os.path.exists(path))
		self.data = []
		with open(path) as f:
			read_first_episode = True
			line = f.readline() #make everything lowercase so indexer works correctly
			episode = []
			while line != "":
				line_num, _, text = line.lower().strip().partition(' ')

				if int(line_num) == 1 and not read_first_episode:
					self.data.append(episode)
					episode = []
				else:
					read_first_episode = False

				text = text.replace(".", " .").replace("?", " ?").replace("  ", ' ')
				if re.search(r'[\t]', text):
					sent_and_labels = text.split('\t')
					sent = sent_and_labels[0].split(' ')
					sent = [indexer.add_and_get_index(x) for x in sent] + [EOS]
					label = indexer.add_and_get_index(sent_and_labels[1]) #not sure what to do with the line numbers
					#could potentially have 2 outputs
					#where the second produces line numbers
					episode.append((sent, label))
					self.max_len = max(self.max_len, len(sent))

				else:
					sent = text.split(' ')
					sent = [indexer.add_and_get_index(x) for x in sent] + [EOS]
					episode.append((sent, None)) #query is dependent upon position in episode so don't know how to separate them
					self.max_len = max(self.max_len, len(sent))

				line = f.readline()

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)