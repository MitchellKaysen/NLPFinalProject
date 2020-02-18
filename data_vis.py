#just messing around with win32console

import win32console
import msvcrt
import numpy as np
import math
from datasets import *
from utils import *
import argparse
import ctypes

def coord(x,y):
	return win32console.PyCOORDType(x,y)

class Column(object):
	def __init__(self, rows, col_idx):
		self.col = col_idx
		self.chars = ' ░▒▓'
		self.counts = np.zeros(rows)
		self.total = 0
		self.rows = rows
	def render(self,buf):
		for r in range(self.rows):
			char = self.chars[int(self.counts[r])]
			col = self.col + 1
			buf.WriteConsoleOutputCharacter(char, coord(col, self.rows - r))
		return
	def density_color_map(self,pct):
		return math.floor(pct / 16.667)

	def add_point(self,val):
		row = int(math.floor(val * self.rows))
		self.counts[:row + 1] += 1
		self.total+= 1

def _parse_args():
    parser = argparse.ArgumentParser(description='data_vis.py')
    parser.add_argument('--persona', type=str, default='self', help='Personas to condition on (both, self, other, none)')
    parser.add_argument('--revision', type=str, default='original', help='Whether to use original or revised dataset')
    return parser.parse_args()

def main():
	args = _parse_args()
	print(args)
	indexer = Indexer()
	train_data = PersonaDataset('train', args.persona, args.revision, indexer)
	valid_data = PersonaDataset('valid', args.persona, args.revision, indexer)
	test_data = PersonaDataset('test', args.persona, args.revision, indexer)
	counts = train_data.counts + valid_data.counts + test_data.counts
	counts = list(counts.values())
	counts.sort(reverse=True)
	total = sum(counts)
	max_val = counts[0]
	max_p = max_val / total
	rows = 30
	cols = 100
	items_per_col = 204
	columns = [Column(rows, i) for i in range(cols)]
	i = 0

	while i < 300:
		val = counts[i]
		idx = int(math.floor(i / 3))
		columns[idx].add_point((val/ total) / (max_p + .00001))
		i+=1

	sc_buf = win32console.CreateConsoleScreenBuffer()
	sc_buf.SetConsoleActiveScreenBuffer()
	# print(type(s))
	print(sc_buf.GetConsoleScreenBufferInfo())
	for col in columns:
		col.render(sc_buf)
	# sc_buf.SetConsoleScreenBufferSize(coord(120, 120))

	sc_buf.WriteConsoleOutputCharacter("data visualization for 300 most frequent tokens", win32console.PyCOORDType(0,0)) 

	c = msvcrt.getwch()
	# sc_buf.SetConsoleScreenBufferSize(s)
	
	# s = coord(100, 100)
	# sc_buf.SetConsoleScreenBufferSize(s)

	sc_buf.Close()
	return

if __name__ == '__main__':
	main()