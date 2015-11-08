import sys
import neural_network_matrix as nn_matrix
import train
import numpy as np
try:
	import cPickle as pickle
except:
	import pickle

files_location = "AUTHORS/"
train_rounds = 200
train_range = 10
min_verify_range = 10
max_verify_range = 20
min_test_range = 10
max_test_range = 20
learn_rate = .1
dummy_var = 1
nlayers = [100, 50, 25]

class authors:

	def __init__(self, load):
		self.lastPercent = 0.0
		self.bestPercent = 0.0
		if (load != '0'):
			print("\ncreating train data...")
			sys.stdout.flush()
			self.td = train.train_data(files_location)
			pickle.dump(self.td, open("_train_data.p", "wb"))
			print("created sucessfully\n")
			sys.stdout.flush()
		else:
			print("\nloading train data...")
			sys.stdout.flush()
			self.td = pickle.load(open("_train_data.p", "rb"))
			print("loaded sucessfully\n")
			sys.stdout.flush()

		self.nn = nn_matrix.neural_network(len(self.td.inputs[0][0]),len(self.td.authors),nlayers,dummy_var, learn_rate)


	def verify(self):
		self.nn.save("lastNeuralNetwork.p")
		total_wrong = 0.0
		total_tests = 0.0
		lp = self.lastPercent
		nn_output = self.nn.output
		inp = self.td.inputs
		for i in range(min_verify_range, max_verify_range):
			for j in range(0, len(inp)):
				if (i < len(self.td.inputs[j])):
					total_tests += 1
					total_wrong += self.td.test_np(nn_output(inp[j][i]), j, i)
		currentPercent = 1 - (total_wrong/total_tests)
		if (self.bestPercent < currentPercent):
			self.nn.save("bestNeuralNetwork.p")
			self.bestPercent = currentPercent
		self.lastPercent = currentPercent
		if (lp > currentPercent):
			if (lp > 1):
				return True
			return False
		else:
			return False

	def best(self):
		total_wrong = 0.0
		total_tests = 0.0
		nn_output = self.nn.output
		inp = self.td.inputs
		for i in range(min_verify_range, max_verify_range):
			for j in range(0, len(inp)):
				if (i < len(self.td.inputs[j])):
					total_tests += 1
					total_wrong += self.td.test_np(nn_output(inp[j][i]), j, i)
		currentPercent = 1 - (total_wrong/total_tests)
		if (self.bestPercent < currentPercent):
			self.nn.save("bestNeuralNetwork.p")
			self.bestPercent = currentPercent

	def progress(self, x, train_rounds):
		sys.stdout.write('\r')
		sys.stdout.write("[%-50s] %d%% %f%%" % ('='*(((x+1)*50)//train_rounds), ((x+1)*100)/train_rounds, 100*self.lastPercent))
		if(self.verify()):
			# self.nn = nn_matrix.neural_network.load("lastNeuralNetwork.p")
			sys.stdout.write("\nNot improving anymore at test: %d" % (x))
			sys.stdout.flush()
			return True
		else:
			sys.stdout.flush()
			return False

	def run(self):
		print("\ntraining...")
		sys.stdout.flush()
		inp = self.td.inputs
		des = self.td.desired
		nn_out_prop = self.nn.out_prop
		for x in range(0, train_rounds):
			if ((x+1)%(train_rounds//50) == 0):
				if (self.progress(x, train_rounds)):
					break
				else:
					self.verify()
			for i in range (0, train_range):
				for j in range(0,len(inp)):
					if (i < len(inp[j])):
						nn_out_prop(inp[j][i], des[j])

		# self.nn = nn_matrix.neural_network.load("bestNeuralNetwork.p")

		print("\nfinished training")
		sys.stdout.flush()

		print("\ntesting...")
		sys.stdout.flush()
		total_wrong = 0
		total_tests = 0
		nn_output = self.nn.output
		for i in range(min_test_range, max_test_range):
			for j in range(0, len(inp)):
				if (i < len(inp[j])):
					total_tests += 1
					total_wrong += self.td.test(nn_output(inp[j][i]), j, i)

		nlayers.insert(0, len(inp[0][0]))
		nlayers.append(len(self.td.authors))
		print("\ncorrect percentage:	", (1 - total_wrong/total_tests) * 100)
		print("number wrong:		", total_wrong)
		print("number of tests:	", total_tests)
		print("traing rounds:		", train_rounds)
		print("train range:		", train_range)
		print("test range:		", min_test_range, "to",max_test_range)
		print("learning rate:		", learn_rate)
		print("neural network:		", nlayers)


def main():
	auth = authors(sys.argv[1])
	auth.run()

main()