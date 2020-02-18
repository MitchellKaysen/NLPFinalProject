import torch
import torch.nn as nn
from torch import optim
import numpy as np 
import torch.nn.functional as F
import re
import torch.utils.data as data
from utils import *
import math

class Attention(nn.Module):
	def __init__(self, input_size):
		super(Attention, self).__init__()
		self.input_size = input_size
		self.fc1 = nn.Linear(input_size, input_size)
		self.fc2 = nn.Linear(input_size, input_size)
		self.fc3 = nn.Linear(input_size, input_size)
		self.relu = nn.LeakyReLU(.1)
		self.drop = nn.Dropout(.5)
	def forward(self,x, qs):
		#q -> b_0 x seq_len_0 x d
		#x -> b_0 x seq_len_1 x d
		#score should be m x n
		# (b x n) x d -> (b x n) x d
		qs = self.drop(torch.sigmoid(self.fc1(qs)))
		# ks = x
		# vs = x
		ks = self.drop(torch.sigmoid(self.fc2(x)))
		vs = self.drop(self.relu(self.fc3(x)))
		# ks = torch.sigmoid(x)
		#(n x d, d x m) -> n x m)
		score = F.softmax(torch.matmul(qs, ks.transpose(1,2)) / np.sqrt(self.input_size), dim=2)
		return torch.matmul(score, vs)
class SelfAttention(nn.Module):
	def __init__(self, input_size):
		super(SelfAttention, self).__init__()
		self.attn = Attention(input_size)
	def forward(self, x):
		return self.attn(x, x)

class PosMask(nn.Module):
	def __init__(self, max_len, emb_dim):
		super(PosMask, self).__init__()
		# f = lambda pos, i : (pos / (10000 ** (2 * i / emb_dim)))
		# arr = [[ math.sin(f(pos, i)) if i % 2 ==0 else math.cos(f(pos, i - 1)) for i in range(emb_dim)] for pos in range(max_len)]
		# self.mtx = torch.from_numpy(np.array(arr)).float()
		# self.mtx.requires_grad_(False)
		self.mtx = torch.randn(max_len, emb_dim)
		self.d = emb_dim

	def forward(self, x):
		x = x.squeeze()
		x = x.unsqueeze(0)
		if (len(x.shape) < 3):
			x = x[None,:]
		
		B, H, _ = x.shape

		return self.mtx[:H, :]

class SimpleEncoder(nn.Module):
	def __init__(self, max_len, block_size, indexer, embedding):
		super(SimpleEncoder, self).__init__()
		self.pe = PosMask(max_len, block_size)

	def forward(self, x_t):
		x_t = x_t.squeeze()
		pe = self.pe(x_t)
		x_t = pe * x_t
		# x_t = self.attn(x_t)
		x_t = torch.sum(x_t, 0)
		return x_t
		
class EntNet(nn.Module):
	def __init__(self, num_blocks, block_size):
		super(EntNet, self).__init__()

		#define properties and intialize hidden state and key values
		self.nblocks = num_blocks
		self.block_size = block_size
		self.keys = nn.Parameter(torch.zeros(1,  num_blocks, block_size).normal_(0, 1/np.sqrt(block_size)))
		#forward layers
		self.nonlin = nn.LeakyReLU(.1)
		self.fc_keys = nn.Linear(block_size, block_size) #this can also be key size im sure
		self.fc_hidden = nn.Linear(block_size, block_size)
		self.fc_input = nn.Linear(block_size, block_size)
		self.drop = nn.Dropout(.4)
		self.attn = Attention(block_size)
		self.layernorm = nn.LayerNorm(block_size)
	#forward pass for single input
	#hidden should be batch, size I think
	#input needs to be B, N, H?
	def forward(self, x_t, hidden=None):
		
		if(len(x_t.shape) < 3) :
			print(x_t.shape)
			raise Exception("Needs to be of dimension B, N, H")
		if hidden is not None:
			block_hiddens = hidden
		else:
			block_hiddens = torch.zeros((x_t.shape[0], self.nblocks,self.block_size )).contiguous().cuda()

		block_keys = self.keys
		tmp = block_hiddens + block_keys
		w = self.attn(x_t, tmp)
		
		new_hiddens = block_hiddens + w
		del block_hiddens
		# del g
		# del gate
		# del u
		# del v
		del w
		del tmp
		return self.layernorm(new_hiddens)

class Babi(nn.Module):
	def __init__(self, **kwargs):
		super(Babi, self).__init__()
		block_size = kwargs.get('block_size', 40)
		num_blocks = kwargs.get('num_blocks', 20)
		embedding = kwargs.get('embedding', None)
		indexer = kwargs.get('indexer', None)
		assert(indexer is not None)
		max_len = kwargs.get('max_len', 20)

		self.embed = nn.Embedding(num_embeddings = len(indexer), embedding_dim = block_size)
		if embedding is not None:
			self.embed.weight.data.copy_(torch.from_numpy(embedding))
		else:
			self.embed._parameters['weight'].data.normal_(0.0, 0.1)

		self.enc = SimpleEncoder(max_len, block_size, indexer, embedding)
		self.dec = SimpleDecoder(indexer, block_size)
		self.mem = EntNet(num_blocks, block_size)
		self.hidden = None

	def reset(self):
		self.mem.clear_mem()
		self.mem.detach_mem()

	def forward(self, s_t):
		seq_len = len(s_t)
		x_t = torch.from_numpy(np.array(s_t)).long().detach()
		x_t = self.embed(x_t)
		self.hidden = self.mem(self.enc(x_t))
		return 

	def query(self, q_t):
		x_t = torch.from_numpy(np.array(q_t)).long().detach()
		x_t = self.embed(x_t).squeeze()
		q = self.enc(x_t)
		return self.dec(q, self.hidden)

class PersonaModelAlt(nn.Module):

	def __init__(self, **kwargs):
		super(PersonaModelAlt, self).__init__()
		block_size = kwargs.get('block_size', 100)
		emb_dim = kwargs.get('emb_dim', 50)
		num_LTM = kwargs.get('num_LTM', 5)
		embedding = kwargs.get('embedding', None)
		indexer = kwargs.get('indexer', None)
		assert(indexer is not None)
		self.block_size = block_size
		self.num_LTM = num_LTM
		self.embedding = embedding
		self.indexer = indexer
		
		self.embed = nn.Embedding(num_embeddings = len(indexer), embedding_dim = emb_dim)
		if embedding is not None:
			self.embed.weight.data.copy_(torch.from_numpy(embedding))
		else:
			self.embed._parameters['weight'].data.normal_(0.0, 1/np.sqrt(emb_dim))

		self.long_mem = EntNet(num_LTM, block_size)
		self.query_module_ltm = Attention(block_size)

		self.layernorm1 = nn.LayerNorm(emb_dim)
		self.layernorm2 = nn.LayerNorm(block_size)
		self.dropout = nn.Dropout(.5)
		self.LTM_state = None
		self.attn = Attention(block_size)
		self.self_attn = SelfAttention(block_size)
		self.encoder = nn.GRU(input_size=emb_dim, hidden_size=block_size, batch_first=True)
		self.decoder = nn.GRU(input_size=emb_dim, hidden_size=block_size,batch_first=True) #produces queries of size e_d


		self.reverse_embed = nn.AdaptiveLogSoftmaxWithLoss(block_size, len(indexer), cutoffs=[7, 21, 500])
		self.nonlin = nn.LeakyReLU(.1)
		self.decoderAUX = nn.GRU(input_size=emb_dim, hidden_size=block_size, batch_first=True)
		self.idxs  = np.arange(len(indexer))
	
	def reset(self):
		# self.work_mem.keys.detach()
		# self.long_mem.keys.detach()
		if(self.LTM_state is not None):
			self.LTM_state.detach()

		del self.LTM_state 
		self.LTM_state = None


	def encode_persona(self, persona):
		u_0 = None
		self.LTM_state = None

		tmp = self.dropout(self.layernorm1(self.embed(persona)))
		# print(torch.cuda.memory_allocated())
		p, u_0 = self.decoder(tmp)
		p = self.layernorm2(self.self_attn(self.nonlin(p)))
		self.LTM_state = self.dropout(self.long_mem(self.dropout(p), self.LTM_state))
		if self.training:
			qs, _ = self.decoderAUX(tmp[:,:-1,:])
			reconst = self.dropout(self.layernorm2(qs + self.dropout(self.query_module_ltm(self.LTM_state,qs))))
			reconst = reconst.contiguous().view(-1, self.block_size)
			labels = persona[:,1:].contiguous().view(-1)
			_, loss = self.reverse_embed(self.nonlin(reconst), labels)
			loss.backward(retain_graph=True)
		return u_0

	def forward(self, persona, utt, y, labels):

		y = self.dropout(self.layernorm1(self.embed(y)))
		utt = self.dropout(self.layernorm1(self.embed(utt)))
		batch_size = utt.shape[0]
		seq_len = y.shape[1]
		hidden_size = utt.shape[2]

		u_0 = self.encode_persona(persona)
		enc_h, h_0 = self.encoder(utt, self.dropout(u_0))
		enc_h = self.nonlin(enc_h)
		enc_h = self.dropout(self.layernorm2(enc_h + self.self_attn(enc_h)))
		enc_h = self.dropout(self.layernorm2(enc_h + self.query_module_ltm(self.LTM_state, enc_h)))
		qs , _ = self.decoder(y, self.dropout(h_0))
		qs = self.dropout(self.layernorm2(self.nonlin(qs)))
		del h_0
		assert(self.LTM_state is not None)
		o_ltm = self.dropout(self.query_module_ltm(self.LTM_state, qs))
		qs = self.dropout(self.layernorm2(qs + self.dropout(self.attn(enc_h, qs)) + o_ltm))
		# out = torch.cat((qs,
		#                  o_wm,
		#                  o_ltm), 2)
		out = self.nonlin(qs).view(-1, self.block_size	)
		outs, loss = self.reverse_embed(out, labels.view(-1))
		return outs.view(batch_size, seq_len), loss

	def predict(self, persona, utt, max_len	):

		u_0 = self.encode_persona(persona)
		utt = self.embed(utt)
		enc_h, h_prev = self.encoder(utt, u_0)
		enc_h = self.layernorm2(enc_h + self.self_attn(self.nonlin(enc_h)))
		enc_h = self.layernorm2(enc_h + self.query_module_ltm(self.LTM_state,enc_h))

		end = self.indexer.index_of('<eos>')

		idx = self.indexer.index_of('<s>')
		output = []
		k = 0
		while idx != end and k < max_len:
			y = torch.tensor([[idx]]).long().cuda()
			y = self.layernorm1(self.embed(y))
			qs, h_prev = self.decoder(y, h_prev)
			qs = self.layernorm2(self.nonlin(qs))
			o_ltm = self.query_module_ltm(self.LTM_state, qs)
			qs = self.layernorm2(qs + self.attn(enc_h, qs) + o_ltm)
			out = self.nonlin(qs).view(-1, self.block_size)
			log_prob = self.reverse_embed.log_prob(out)
			idx = int(torch.argmax(log_prob, dim=1))
			output.append(idx)
			k+=1
		return output

class Seq2SeqAlt(nn.Module):

	def __init__(self, **kwargs):
		super(Seq2SeqAlt, self).__init__()
		block_size = kwargs.get('block_size', 100)
		emb_dim = kwargs.get('emb_dim', 50)
		num_WM = kwargs.get('num_WM', 5)
		num_LTM = kwargs.get('num_LTM', 5)
		embedding = kwargs.get('embedding', None)
		indexer = kwargs.get('indexer', None)
		assert(indexer is not None)
		self.block_size = block_size
		self.num_WM = num_WM
		self.num_LTM = num_LTM
		self.embedding = embedding
		self.indexer = indexer
		
		self.embed = nn.Embedding(num_embeddings = len(indexer), embedding_dim = emb_dim)
		if embedding is not None:
			self.embed.weight.data.copy_(torch.from_numpy(embedding))
		else:
			self.embed._parameters['weight'].data.normal_(0.0, 1/np.sqrt(emb_dim))
		self.layernorm1 = nn.LayerNorm(emb_dim)
		self.layernorm2 = nn.LayerNorm(block_size)
		self.attn = Attention(block_size)
		self.self_attn = SelfAttention(block_size)
		self.encoder = nn.GRU(input_size=emb_dim, hidden_size=block_size, batch_first=True)
		self.drop = nn.Dropout(.5)
		self.reverse_embed = nn.AdaptiveLogSoftmaxWithLoss(block_size, len(indexer), cutoffs=[7, 21])
		self.nonlin = nn.LeakyReLU(.1)

		self.decoder = nn.GRU(input_size=emb_dim, hidden_size=block_size,batch_first=True) #produces queries of size e_d
		self.idxs  = np.arange(len(indexer))
	
	def reset(self):
		# self.work_mem.keys.detach()
		# self.long_mem.keys.detach()
		return

	def encode_persona(self, persona):
		return 

	def forward(self, persona, utt, y, labels):

		utt = torch.cat((persona, utt), dim=1)
		y = self.drop(self.layernorm1(self.embed(y)))
		utt = self.drop(self.layernorm1(self.embed(utt)))
		batch_size = utt.shape[0]
		seq_len = y.shape[1]
		hidden_size = utt.shape[2]

		enc_h, h_0 = self.encoder(utt)
		enc_h = self.drop(enc_h)
		enc_h = self.layernorm2(enc_h + self.self_attn(enc_h))
		qs , u_next = self.decoder(y, h_0)
		qs = self.drop(qs)
		qs = self.layernorm2(qs + self.attn(enc_h, qs))
		# out = torch.cat((qs,
		#                  o_wm,
		#                  o_ltm), 2)
		out = self.nonlin(qs).view(-1, self.block_size	)
		out, loss = self.reverse_embed(out, labels.view(-1))
		return  out.view(batch_size, seq_len),  loss

	def predict(self, persona, utt, max_len	):

		utt = torch.cat((persona, utt), dim=1)
		utt = self.layernorm1(self.embed(utt))
		enc_h, h_prev = self.encoder(utt)
		enc_h = self.layernorm2(enc_h + self.self_attn(enc_h))
		end = self.indexer.index_of('<eos>')

		idx = self.indexer.index_of('<s>')
		output = []
		k = 0
		while idx != end and k < max_len:
			y = torch.tensor([[idx]]).long().cuda()
			y = self.layernorm1(self.embed(y))
			q, h_prev = self.decoder(y, h_prev)
			q = self.layernorm2(q + self.attn(enc_h,q))
			out = self.nonlin(q).view(-1, self.block_size)
			log_prob = self.reverse_embed.log_prob(out)
			idx = int(torch.argmax(log_prob, dim=1))
			output.append(idx)
			k+=1
		return output

class Seq2SeqAltNoAttn(nn.Module):

	def __init__(self, **kwargs):
		super(Seq2SeqAltNoAttn, self).__init__()
		block_size = kwargs.get('block_size', 100)
		emb_dim = kwargs.get('emb_dim', 50)
		num_WM = kwargs.get('num_WM', 5)
		num_LTM = kwargs.get('num_LTM', 5)
		embedding = kwargs.get('embedding', None)
		indexer = kwargs.get('indexer', None)
		assert(indexer is not None)
		self.block_size = block_size
		self.num_WM = num_WM
		self.num_LTM = num_LTM
		self.embedding = embedding
		self.indexer = indexer
		
		self.embed = nn.Embedding(num_embeddings = len(indexer), embedding_dim = emb_dim)
		if embedding is not None:
			self.embed.weight.data.copy_(torch.from_numpy(embedding))
		else:
			self.embed._parameters['weight'].data.normal_(0.0, 1/np.sqrt(emb_dim))
		self.drop = nn.Dropout(.5)

		self.layernorm1 = nn.LayerNorm(emb_dim)
		self.layernorm2 = nn.LayerNorm(block_size)
		self.reverse_embed = nn.AdaptiveLogSoftmaxWithLoss(block_size, len(indexer), cutoffs=[7, 21, 500])
		self.nonlin = nn.LeakyReLU(.1)
		self.encoder = nn.GRU(input_size=emb_dim, hidden_size=block_size, batch_first=True)
		self.decoder = nn.GRU(input_size=emb_dim, hidden_size=block_size,batch_first=True) #produces queries of size e_d
		self.idxs  = np.arange(len(indexer))
	
	def reset(self):
		# self.work_mem.keys.detach()
		# self.long_mem.keys.detach()
		return

	def encode_persona(self, persona):
		return 

	def forward(self, persona, utt, y, labels):

		utt = torch.cat((persona, utt), dim=1)
		y = self.drop(self.layernorm1(self.embed(y)))
		utt = self.drop(self.layernorm1(self.embed(utt)))
		batch_size = utt.shape[0]
		seq_len = y.shape[1]
		hidden_size = utt.shape[2]

		enc_h, h_0 = self.encoder(utt)
		qs , u_next = self.decoder(y, self.drop(h_0))
		qs = self.drop(self.layernorm2(qs))
		# out = torch.cat((qs,
		#                  o_wm,
		#                  o_ltm), 2)
		out = self.nonlin(qs).view(-1, self.block_size	)
		out, loss = self.reverse_embed(out, labels.view(-1))
		return  out.view(batch_size, seq_len), loss

	def predict(self, persona, utt, max_len	):

		utt = torch.cat((persona, utt), dim=1)
		utt = self.layernorm1(self.embed(utt))
		_, h_prev = self.encoder(utt)
		end = self.indexer.index_of('<eos>')

		idx = self.indexer.index_of('<s>')
		output = []
		k = 0
		while idx != end and k < max_len:
			y = torch.tensor([[idx]]).long().cuda()
			y = self.layernorm1(self.embed(y))
			q, h_prev = self.decoder(y, h_prev)
			q = self.layernorm2(q)
			out = self.nonlin(q).view(-1, self.block_size)
			log_prob = self.reverse_embed.log_prob(out)
			idx = int(torch.argmax(log_prob, dim=1))
			output.append(idx)
			k+=1
		return output

class Seq2SeqBaseline(nn.Module):

	def __init__(self, **kwargs):
		super(Seq2SeqBaseline, self).__init__()
		block_size = kwargs.get('block_size', 100)
		emb_dim = kwargs.get('emb_dim', 50)
		num_WM = kwargs.get('num_WM', 5)
		num_LTM = kwargs.get('num_LTM', 5)
		embedding = kwargs.get('embedding', None)
		indexer = kwargs.get('indexer', None)
		assert(indexer is not None)
		self.block_size = block_size
		self.num_WM = num_WM
		self.num_LTM = num_LTM
		self.embedding = embedding
		self.indexer = indexer
		
		self.embed = nn.Embedding(num_embeddings = len(indexer), embedding_dim = emb_dim)
		if embedding is not None:
			self.embed.weight.data.copy_(torch.from_numpy(embedding))
		else:
			self.embed._parameters['weight'].data.normal_(0.0, 1/np.sqrt(emb_dim))

		self.encoder = nn.GRU(input_size=emb_dim, hidden_size=block_size, batch_first=True)
		self.layernorm1 = nn.LayerNorm(emb_dim)
		self.layernorm2 = nn.LayerNorm(block_size)
		self.drop = nn.Dropout(.5)
		self.reverse_embed = nn.AdaptiveLogSoftmaxWithLoss(block_size, len(indexer), cutoffs=[7, 21, 500])
		self.nonlin = nn.LeakyReLU(.1)

		self.decoder = nn.GRU(input_size=emb_dim, hidden_size=block_size,batch_first=True) #produces queries of size e_d
		self.idxs  = np.arange(len(indexer))
	
	def reset(self):
		# self.work_mem.keys.detach()
		# self.long_mem.keys.detach()
		return

	def encode_persona(self, persona):
		return 

	def forward(self, persona, utt, y, labels):

		y = self.drop(self.layernorm1(self.embed(y)))
		utt = self.drop(self.layernorm1(self.embed(utt)))
		batch_size = utt.shape[0]
		seq_len = y.shape[1]
		hidden_size = utt.shape[2]

		enc_h, h_0 = self.encoder(utt)
		qs , u_next = self.decoder(y, self.drop(h_0))
		qs = self.drop(self.layernorm2(qs))
		# out = torch.cat((qs,
		#                  o_wm,
		#                  o_ltm), 2)
		out = self.nonlin(qs).view(-1, self.block_size	)
		out, loss = self.reverse_embed(out, labels.view(-1))
		return  out.view(batch_size, seq_len), loss

	def predict(self, persona, utt, max_len	):

		utt = self.layernorm1(self.embed(utt))
		enc_h, h_prev = self.encoder(utt)
		end = self.indexer.index_of('<eos>')

		idx = self.indexer.index_of('<s>')
		output = []
		k = 0
		while idx != end and k < max_len:
			y = torch.tensor([[idx]]).long().cuda()
			y = self.layernorm1(self.embed(y))
			q, h_prev = self.decoder(y, h_prev)
			q = self.layernorm2(q)
			out = self.nonlin(q).view(-1, self.block_size)
			log_prob = self.reverse_embed.log_prob(out)
			idx = int(torch.argmax(log_prob, dim=1))
			output.append(idx)
			k+=1
		return output
