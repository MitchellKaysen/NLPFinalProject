import argparse
import os
import torch
import torch.utils.data as data
import gc
import torch.nn as nn
from torch import optim
from utils import *
from models import *
import datasets
from datasets import *
import random
import sacrebleu

def _parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--persona', type=str, default='self', help='Personas to condition on (both, self, other, none)')
    parser.add_argument('--revision', type=str, default='original', help='Whether to use original or revised dataset')
    parser.add_argument('--glove', type=int, default=50, help='GloVe embedding size to use')
    parser.add_argument('--batch_size', type=int, default=4, help="Minibatch size")
    parser.add_argument('--num_LTM', type=int, default=5, help="number of memory cells2")
    parser.add_argument('--block_size', type=int, default=50, help="Size of hidden layers")
    parser.add_argument('--epochs', type=int, default=20, help="Epochs to run")
    parser.add_argument('--model', type=str, default='persona')
    args = parser.parse_args()
    return args

#global dataset length maximums
#returns the maximum utterance length
def _pad_data(datasets):
	max_len_p = 0
	max_len_h = 0
	max_n_sent = 0
	max_n_pers = 0
	for d in datasets:
		max_len_p = max(max_len_p, d.ML['persona'])
		max_len_h = max(max_len_h, d.ML['sent'])
		max_n_sent = max(max_n_sent, d.ML['n_sent'])
		max_n_pers = max(max_n_pers, d.ML['n_pers'])
	for d in datasets:
		d.ML = {
			'persona' : max_len_p,
			'sent' : max_len_h,
			'n_sent' : max_n_sent,
			'n_pers' : max_n_pers
		}

	print(max_n_sent)
	return max_len_h

def _collate_pad(data):
	s_per, s_hist, p_per, p_hist, labels = zip(*data)
	# 	print(x)
	s_per = torch.tensor(list(s_per)).long()
	s_hist = torch.tensor(list(s_hist)).long()
	p_per = torch.tensor(list(p_hist)).long()
	p_hist = torch.tensor(list(p_hist)).long()
	labels = torch.tensor(list(labels)).long()
	return {
		's_per' : s_per,
		's_hist' : s_hist,
		'p_per' : p_per,
		'p_hist' : p_hist,
		'labels' : labels
	}

def _collate_pad_alt(indexer, data):
	personas, thems, yous = zip(*data)
	
	ps = [p['self'] for p in personas]
	max_len = max([len(p) for p in ps])
	pad_idx = indexer.index_of('<pad>')
	pad = [pad_idx for _ in range(max_len)]
	
	ps = [[indexer.index_of(w) for w in p] for p in ps]
	personas = [p + pad[len(p):] for p in ps]
	personas = torch.tensor(personas).long()

	max_len = max(max([len(t) for t in thems]), max([len(y) for y in yous]) - 1)
	pad = [pad_idx for _ in range(max_len)]
	
	thems = [[indexer.index_of(w) for w in t] for t in thems]
	thems = [t + pad[len(t):] for t in thems]
	thems = torch.tensor(thems).long()

	yous = [[indexer.index_of(w) for w in t] for t in yous]
	labels = [y[1:] for y in yous]	
	labels = [t + pad[len(t):] for t in labels]
	labels = torch.tensor(labels).long()

	yous = [y[:-1] for y in yous]
	yous = [t + pad[len(t):] for t in yous]
	yous = torch.tensor(yous).long()

	return personas, thems, yous, labels

def _losses(lossfn, y_preds, y_actual):
	y_actual = [torch.from_numpy(np.array(y)).long() for y in y_actual]
	y_actual = torch.cat(y_actual) 
	# for i in range(len(y_pred)):
	# 	y = y_actual[i]
	# 	y = torch.from_numpy(np.array(y)).long()
	# 	y_hat = y_preds[i].squeeze(0)
	# 	print(y_hat.shape)
	# 	print(y.shape)
	# 	l_val = lossfn(y_hat, y)
	# 	losses.append(l_val)
	return lossfn(y_preds, y_actual)

def sampler(len_data, batch_size, drop_last=False):
	while True:
		idxs = random.sample(range(len_data), len_data)
		loc = 0
		batch_slice = loc + batch_size
		while loc < len_data:

			batch_slice = min(len_data, batch_slice)

			yield idxs[loc:batch_slice]

			loc = batch_slice
			batch_slice = loc + batch_size

def train_persona_full(args):
	batch_size = args.batch_size
	emb_dim = args.glove
	num_WM = args.num_WM
	num_LTM = args.num_LTM
	persona = args.persona
	revision = args.revision
	indexer = Indexer()

	cuda = torch.cuda.current_device()
	print(torch.cuda.get_device_name(cuda))
	model_save_path = "models/persona_"+persona+"-"+revision +"_"+str(emb_dim) +"d_"+str(num_WM)+"-"+str(num_LTM)+"b.pth"
	print(args)
	print("Loading datasets...")
	train_data = PersonaDataset('train', args.persona, args.revision, indexer)
	print("%d examples" % len(train_data))
	print("%d" % train_data.ML['persona'])
	valid_data = PersonaDataset('valid', args.persona, args.revision, indexer)
	print("%d examples" % len(valid_data))

	test_data = PersonaDataset('test', args.persona, args.revision, indexer)
	print("%d examples" % len(test_data))

	counts = train_data.counts + valid_data.counts + test_data.counts
	#Zipf's law for class weights, s==1
	vals = counts.values()
	print(counts)
	l = .4

	freq = np.array(list(vals))
	
	keys = np.array(list(counts.keys()))
	idxs = np.argsort(keys)
	term_freq = freq[idxs] # sort frequencies by class index
	
	class_weights = (1 / np.power(term_freq, l))
	class_weights = torch.from_numpy(class_weights).float()
	max_len = _pad_data([train_data, valid_data, test_data])

	# train_loader = data.DataLoader(train_data,
	                               # batch_size=batch_size,
	                               # shuffle=False,
	                               # collate_fn=_collate_pad)
	# valid_loader = data.DataLoader(valid_data,
	#                                batch_size=4,
	#                                shuffle=False,
	#                                collate_fn=_collate_pad)
	# test_loader = data.DataLoader(test_data,
	#                               batch_size=4,
	#                               shuffle=False)
	print("...Done")
	sampler = data.BatchSampler(data.RandomSampler(train_data, replacement=True, num_samples=2000), 
	                            batch_size=batch_size,
	                            drop_last=False)
	valid_sampler = data.BatchSampler(data.RandomSampler(valid_data, replacement=True, num_samples=200),
	                                	batch_size=4,
	                                	drop_last=False)
	train_loader = data.DataLoader(train_data,
	                               batch_sampler=sampler,
	                               collate_fn=_collate_pad)
	valid_loader = data.DataLoader(valid_data,
	                               batch_sampler=valid_sampler,
	                               collate_fn=_collate_pad)
	print("%d tokens" % len(indexer))
	print("<s>: %d | <eos>: %d | <pad>: %d" %(indexer.index_of('<s>'), indexer.index_of('<eos>'), indexer.index_of('<pad>')))
	pad_idx = indexer.index_of('<pad>')
	if batch_size > len(train_data):
		batch_size = 32
	print("Loading GLoVe...")
	#NLL loss during training
	emb_weights = None

	glove = datasets.load_glove(emb_dim)
	num_words = len(indexer)
	emb_weights = np.zeros((num_words, emb_dim))
	for i in range(num_words):
		word = indexer.get_object(i)
		e = glove.get(word)
		if e is not None:
			emb_weights[i] = e
		else:
			emb_weights[i] = np.random.normal(scale=(1 /np.sqrt(emb_dim)), size=(emb_dim,)) #Xavier norm
	print("...Done")
	epochs = args.epochs
	batch_size = args.batch_size
	block_size = args.block_size
	# model = PersonaModel( num_WM=num_WM, num_LTM=num_LTM, emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=emb_weights)
	model = PersonaModel( num_WM=num_WM, num_LTM=num_LTM, emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=emb_weights)
	model.cuda()
	loss = nn.CrossEntropyLoss(ignore_index=pad_idx)#weight=class_weights)

	optimizer = optim.AdamW(model.parameters(), lr=1e-3)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
	trainer = PersonaClassifier(model,block_size, max_len, num_WM)
	for epoch in range(epochs):
		model.train()
		loss_vals = []

		print("EPOCH %d" % epoch)
		k = 0
		for batch in iter(train_loader):
			# print(k)
			if k % 25 == 0:
				print(".", end='')
			optimizer.zero_grad()
			model.reset()
			print(torch.cuda.memory_allocated())
			# print(torch.cuda.memory_cached())
			# print("**********")
			persona = batch['s_per'].cuda()
			p_hist = batch['p_hist'].cuda()
			s_hist = batch['s_hist'].cuda()
			y_pred = trainer.forward(persona, p_hist, s_hist)
			y_pred = y_pred.permute(0,3,1,2)
			# print(y_pred.is_cuda)
			train_loss_val = loss(y_pred, batch['labels'].cuda())
			print(torch.cuda.memory_allocated())

			train_loss_val.backward()
			print(torch.cuda.memory_allocated())
			# print()
			for param in list(filter(lambda p : p.grad is not None, model.parameters())):
				param.grad.add_(torch.randn(param.size()).cuda() * 0.01)
			loss_vals.append(train_loss_val.item())
			torch.nn.utils.clip_grad_norm_(model.parameters(),40)

			del persona
			del p_hist
			del s_hist
			optimizer.step()
			del y_pred
			del train_loss_val
			del batch
			
			# print(torch.cuda.memory_cached())
			# print("**********")
			torch.cuda.empty_cache()

			torch.cuda.ipc_collect()
			torch.cuda.synchronize()
			k+= 1
		loss_avg = sum(loss_vals) / len(loss_vals)
		print("\ntrain loss: %f"  % loss_avg)
		# assert(False)

		scheduler.step()
		with torch.no_grad():
			model.eval()
			valid_loss_val = 0
			k = 0
			for batch in iter(valid_loader):
				if k % 25 == 0:
					print(".", end='')
				persona = batch['s_per']
				p_hist = batch['p_hist']
				s_hist = batch['s_hist']
				y_pred = trainer.forward(persona, p_hist, s_hist)
				y_pred = y_pred.permute(0,3,1,2)
				# print(y_pred.is_cuda)
				valid_loss_val += loss(y_pred, batch['labels'].cuda()).item()
				del persona
				del p_hist
				del s_hist
				del y_pred
				del batch
				k+=1
			valid_loss_val = valid_loss_val / k
			print("Valid loss: %f" % valid_loss_val)

	save_params = {
		'batch_size': batch_size,
		'emb_dim' : emb_dim,
		'num_WM' : num_WM,
		'num_LTM' : num_LTM,
		'persona' : args.persona,
		'revision' : revision,
		'max_len' : max_len
	}
	torch.save({
	        	'indexer': indexer,
	        	'param_info' : save_params,
	        	'model_state_dict': model.state_dict()}, model_save_path)

def train_persona_alt(args):
	batch_size = args.batch_size
	emb_dim = args.glove
	num_LTM = args.num_LTM
	persona = args.persona
	revision = args.revision
	block_size = args.block_size
	

	cuda = torch.cuda.current_device()
	print(torch.cuda.get_device_name(cuda))
	

	print(args)
	print("Loading datasets...")
	train_data = PersonaDatasetAlt('train', args.persona, args.revision)
	print("%d examples" % len(train_data))
	valid_data = PersonaDatasetAlt('valid', args.persona, args.revision)
	print("%d examples" % len(valid_data))

	test_data = PersonaDatasetAlt('test', args.persona, args.revision)
	print("%d examples" % len(test_data))

	counts = train_data.counts + valid_data.counts + test_data.counts
	#Zipf's law for class weights, s==1

	# print(counts)
	counts = counts.most_common()
	# vals = counts.values()
	l = .4
	indexer = Indexer()
	indexer.add_and_get_index('<pad>')
	idxs = [indexer.add_and_get_index(k) for k, v in counts]
	# keys = np.array(list(counts.keys()))
	# idxs = np.argsort(keys)
	# term_freq = freq[idxs] # sort frequencies by class index
	
	# class_weights = (1 / np.power(term_freq, l))
	# class_weights = torch.from_numpy(class_weights).float()
	max_len = max(train_data.max_len_h, valid_data.max_len_h, test_data.max_len_h)
	#max length of sentence only necessary for prediction time
	collate_fn = lambda data :  _collate_pad_alt(indexer, data)
	print("...Done")
	sampler = data.BatchSampler(data.RandomSampler(train_data), 
	                            batch_size=batch_size,
	                            drop_last=False)
	valid_sampler = data.BatchSampler(data.RandomSampler(valid_data),
	                                	batch_size=batch_size,
	                                	drop_last=False)
	train_loader = data.DataLoader(train_data,
	                               batch_sampler=sampler,
	                               collate_fn=collate_fn)
	valid_loader = data.DataLoader(valid_data,
	                               batch_sampler=valid_sampler,
	                               collate_fn=collate_fn)
	print("%d tokens" % len(indexer))
	print("<s>: %d | <eos>: %d | <pad>: %d" %(indexer.index_of('<s>'), indexer.index_of('<eos>'), indexer.index_of('<pad>')))
	pad_idx = indexer.index_of('<pad>')
	if batch_size > len(train_data):
		batch_size = 32
	print("Loading GLoVe...")
	#NLL loss during training
	emb_weights = None

	glove = datasets.load_glove(emb_dim)
	num_words = len(indexer)
	emb_weights = np.zeros((num_words, emb_dim))
	for i in range(num_words):
		word = indexer.get_object(i)
		e = glove.get(word)
		if e is not None:
			emb_weights[i] = e
		else:
			emb_weights[i] = np.random.normal(scale=(1 /np.sqrt(emb_dim)), size=(emb_dim,)) #Xavier norm
	print("...Done")
	epochs = args.epochs
	batch_size = args.batch_size

	# model = PersonaModel( num_WM=num_WM, num_LTM=num_LTM, emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=emb_weights)
	if(args.model == "persona"):
		model_save_path = "models/persona_ALT_"+persona+"-"+revision +"_"+str(emb_dim) +"e_"+str(block_size) + "b_"+str(num_LTM)+"n.pth"
		model = PersonaModelAlt( num_LTM=num_LTM, emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=emb_weights)

	elif args.model == 'seq2seq':
		model_save_path = "models/seq2seq_ALT_"+persona+"-"+revision +"_"+str(emb_dim) +"e_"+str(block_size)+"b.pth"
		model = Seq2SeqAlt(emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=emb_weights)
	elif(args.model == "s2snoattn"):
		model_save_path = "models/s2snoattn_"+persona+"-"+revision +"_"+str(emb_dim) +"e_"+str(block_size)+"b.pth"

		model = Seq2SeqAltNoAttn(emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=None)
	elif(args.model == "s2sbaseline"):
		model_save_path = "models/s2sbaseline_"+persona+"-"+revision +"_"+str(emb_dim) +"e_"+str(block_size)+"b.pth"

		model = Seq2SeqBaseline( emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=None)
	model.cuda()

	#adaptive softmax does the loss for you so if you want to ignore an index you'll have to pass it in yourself
	# loss = nn.CrossEntropyLoss(ignore_index=pad_idx)#weight=class_weights) 

	optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
	iters = len(train_loader)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult = 2)
	# scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=.5)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True,threshold=.001, patience=5, factor=.5)
	for epoch in range(epochs):
		model.train()
		loss_vals = 0

		print("EPOCH %d" % epoch)
		k = 0
		for i, batch in enumerate(iter(train_loader)):
			# print(k)
			scheduler.step(epoch + i / iters)
			personas, utts, ys, labels = batch
			if k % 25 == 0:
				print(".", end='')
			optimizer.zero_grad()
			model.reset()

			_, loss = model(personas.cuda(), utts.cuda(), ys.cuda(), labels.cuda())
			loss.backward()
			loss_vals += float(loss)
			del loss
			torch.nn.utils.clip_grad_norm_(model.parameters(),20)

			optimizer.step()
			
			#i don't know what this does but I will try anything to reduce memory usage at this point
			torch.cuda.empty_cache()
			torch.cuda.ipc_collect()
			torch.cuda.synchronize()
			k+= 1
		loss_avg = loss_vals / iters
		print("\ntrain loss: %f"  % loss_avg)
		# assert(False)
		scheduler.step()

		with torch.no_grad():
			model.eval()
			valid_loss_val = 0
			k = 0
			for  personas, utts, ys, labels in iter(valid_loader):
				if k % 25 == 0:
					print(".", end='')
				# print(y_pred.is_cuda)
				model.reset()
				_, loss = model(personas.cuda(), utts.cuda(), ys.cuda(), labels.cuda())
				valid_loss_val += float(loss.item())
				del loss
				k+=1
			valid_loss_val = valid_loss_val / k
			print("Valid loss: %f" % valid_loss_val)
		# scheduler.step(valid_loss_val)

	save_params = {
		'batch_size': batch_size,
		'emb_dim' : emb_dim,
		'num_LTM' : num_LTM,
		'persona' : args.persona,
		'revision' : revision,
		'max_len' : max_len,
		'block_size' : block_size,
		'model' : args.model
	}
	torch.save({
	        	'indexer': indexer,
	        	'param_info' : save_params,
	        	'model_state_dict': model.state_dict()}, model_save_path)

if __name__ == '__main__':
	f = open('persona_log.txt', 'w+')
	f.close()
	args = _parse_args()
	train_persona_alt(args)
