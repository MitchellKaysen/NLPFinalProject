from models import *
import torch
import os
import datasets
from datasets import *
import argparse
import sacrebleu

def _parse_args():
    parser = argparse.ArgumentParser(description='eval.py')
    parser.add_argument('--path', type=str, default=None, help="model to evaluate")
    parser.add_argument('--model', type=str, default='persona')

    args = parser.parse_args()
    return args


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

if __name__ == '__main__':
	args = _parse_args()
	print(args)
	path = os.path.join("models/", args.path)
	if not os.path.exists(path):
		raise Exception("No such path")

	model_info = torch.load(path)

	param_info = model_info['param_info']
	batch_size = param_info['batch_size']
	emb_dim = param_info['emb_dim']
	num_LTM = param_info['num_LTM']
	persona = param_info['persona']
	revision = param_info['revision']
	max_len = param_info['max_len']
	block_size = param_info['block_size']
	model_type = param_info['model']
	indexer = model_info['indexer']
	print(param_info)

	pad_idx = indexer.index_of('<pad>')
	print("<pad>: %d" % indexer.index_of('<pad>'))
	print("Loading datasets...")
	test_data = PersonaDatasetAlt('test', persona, revision)
	collate = lambda data : _collate_pad_alt(indexer, data)
	ppl_loader = data.DataLoader(test_data, batch_size=500, shuffle=False, collate_fn=collate)
	test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False,collate_fn=collate	)
	print("...Done")
	print("%d tokens" % len(indexer))
	print("<s>: %d | <eos>: %d | <pad>: %d" %(indexer.index_of('<s>'), indexer.index_of('<eos>'), indexer.index_of('<pad>')))
	if(model_type == "seq2seq"):
		model = Seq2SeqAlt(  emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=None)
	elif(model_type == "persona"):
		model = PersonaModelAlt( num_LTM=num_LTM,  emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=None)
	elif(model_type == "s2snoattn"):
		model = Seq2SeqAltNoAttn( emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=None)
	elif(model_type == "s2sbaseline"):
		model = Seq2SeqBaseline( emb_dim=emb_dim, block_size=block_size, indexer=indexer, embedding=None)
	model.load_state_dict(model_info['model_state_dict'])

	model.eval()
	model.cuda()

	with torch.no_grad():
		ppl = 0
		for i, sample in enumerate(iter(ppl_loader)):
			model.reset()
			persona, utt, y, labels = sample
			y_np = labels.detach().numpy()
			mask = np.where(y_np != 0, 1, 0)
			lens = np.sum(mask, axis=1)
			outs, _ = model(persona.cuda(), utt.cuda(), y.cuda(), labels.cuda())
			outs = outs.detach().cpu().numpy()
			outs = outs * mask
			ppls =  np.exp(-outs.sum(axis=1) / lens)
			ppl += float(ppls.sum())
			if (i + 1)  % 3 == 0:
				batch_ppl = ppls.sum() / int(utt.shape[0])
				print("Avg: %f" % batch_ppl)
		print("Perplexity: %d" % (ppl / len(test_data)))
		refs = []
		sys  = []
		for i, sample  in enumerate(iter(test_loader)):
			model.reset()
		# for i in range(10):
		# 	x = test_data[i]
			# print("****************************")
			persona, utt, _, label = sample
			y_pred = model.predict(persona.cuda(), utt.cuda(), max_len	)
			p , in_sent, label = test_data[i]
			
			out_sent = [indexer.get_object(idx) for idx in y_pred]
			if out_sent[-1] == '<EOS>':
				out_sent = out_sent[:-1]
			y = label[1:-1]
			s_t = ' '.join(in_sent)
			y_hat = ' '.join(out_sent)
			sys.append(y_hat)
			y = ' '.join(y)
			refs.append(y)
			print(' '.join(p['self']))
			print(' '.join(in_sent))

			# print("X: %s" % s_t)
			print("Y: %s | Y_hat %s" % (y,y_hat))
			print(sacrebleu.corpus_bleu([y_hat], [[y]]))
			i+=1

		print(sacrebleu.corpus_bleu(sys, [refs]))
