import neural_network_matrix as nn_matrix
import info_parser as ip
import numpy as np
import sys
import os

class train_data:
	def __init__(self, folder_name):
		self.prsr = ip.parser()
		self.authors = os.listdir(folder_name)
		self.desired = []
		for i in range(0, len(self.authors)):
			temp_des = []
			for j in range(0, len(self.authors)):
				temp_des.append((i == j) * 1)
			self.desired.append(np.array([temp_des]).T)
		self.inputs = []
		self.files = []
		for i in range(0, len(self.authors)):
			author_files = os.listdir(folder_name+self.authors[i]+"/")
			self.files.append(author_files)
			author_input = []
			for afile in author_files:
				author_input.append(np.array([self.prsr.parse(folder_name+self.authors[i]+"/"+afile)]).T)
			self.inputs.append(author_input)
	def test(self, output, j, k):
		o  = output.T.tolist()[0]
		o_max = - 1
		max_i = - 1
		for i in range(0,len(o)):
			if (o[i] > o_max):
				max_i = i
				o_max = o[i]
		if (max_i == j):
			return 0
		else:
			print("outputted", self.authors[max_i], "should be", self.authors[j]+"/"+self.files[j][k])
			sys.stdout.flush()
			return 1
	def test_np(self, output, j, k):
		o  = output.T.tolist()[0]
		o_max = -1
		max_i = -1
		for i in range(0,len(o)):
			if (o[i] > o_max):
				max_i = i
				o_max = o[i]
		if (max_i == j):
			return 0
		else:
			return 1

def main():
	td = train_data("docs2/")



# main()