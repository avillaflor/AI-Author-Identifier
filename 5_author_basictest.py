import neural_network_matrix as nn_matrix
import info_parser as ip
import numpy as np
import sys

def test(output):
	o  = output.T.tolist()[0]
	o_max = 0
	max_i = -1
	for i in range(0,len(o)):
		if (o[i] > o_max):
			max_i = i
			o_max = o[i]
	if (max_i == 0):
		return "shakespeare"
	elif (max_i == 1):
		return  "twain"
	elif (max_i == 2):
		return "dickens"
	elif (max_i == 3):
		return "doyle"
	elif (max_i == 4):
		return "milton"

def main():
	train_rounds = 500
	learn_rate = .5
	shake = "docs2/shakespeare/"
	twain = "docs2/twain/"
	dickens = "docs2/dickens/"
	doyle = "docs2/doyle/"
	milton = "docs2/milton/"
	txt = ".txt"
	nn = nn_matrix.neural_network(10,5,[20],1, learn_rate)
	prsr = ip.parser()
	shake_desired = 	np.array([[1], [0], [0], [0], [0]])
	twain_desired =		np.array([[0], [1], [0], [0], [0]])
	dickens_desired = 	np.array([[0], [0], [1], [0], [0]])
	doyle_desired = 	np.array([[0], [0], [0], [1], [0]])
	milton_desired = 	np.array([[0], [0], [0], [0], [1]])
	input_shake = []
	input_twain = []
	input_dickens = []
	input_doyle = []
	input_milton = []
	for j in range(1, 16):
		input_shake.append(np.array([prsr.parse(shake+str(j)+txt)]).T)
		input_twain.append(np.array([prsr.parse(twain+str(j)+txt)]).T)
		input_dickens.append(np.array([prsr.parse(dickens+str(j)+txt)]).T)
		input_doyle.append(np.array([prsr.parse(doyle+str(j)+txt)]).T)
		input_milton.append(np.array([prsr.parse(milton+str(j)+txt)]).T)


	for x in range(0, train_rounds):
		for i in range(0, 15):
			nn.output(input_shake[i])
			nn.back_prop(shake_desired)
			nn.output(input_twain[i])
			nn.back_prop(twain_desired)
			nn.output(input_dickens[i])
			nn.back_prop(dickens_desired)
			nn.output(input_doyle[i])
			nn.back_prop(doyle_desired)
			nn.output(input_milton[i])
			nn.back_prop(milton_desired)
	total_wrong = 0
	total_tests = 0
	for a in range(16,19):
		total_tests += 5
		stra = str(a)
		test_shake = prsr.parse(shake+stra+txt)
		test_twain = prsr.parse(twain+stra+txt)
		test_dickens = prsr.parse(dickens+stra+txt)
		test_doyle = prsr.parse(doyle+stra+txt)
		test_milton = prsr.parse(milton+stra+txt)
		shake_out = nn.output(test_shake)
		twain_out = nn.output(test_twain)
		dickens_out = nn.output(test_dickens)
		doyle_out = nn.output(test_doyle)
		milton_out = nn.output(test_milton)
		# print("test"+stra+":")
		if (test(shake_out) != "shakespeare"): 
			total_wrong += 1
			print("shake_out"+stra+": ", test(shake_out))
		if (test(twain_out) != "twain"): 
			total_wrong += 1
			print("twain_out"+stra+": ", test(twain_out))
		if (test(dickens_out) != "dickens"): 
			total_wrong += 1
			print("dikns_out"+stra+": ", test(dickens_out))
		if (test(doyle_out) != "doyle"): 
			total_wrong += 1
			print("doyle_out"+stra+": ", test(doyle_out))
		if (test(milton_out) != "milton"): 
			total_wrong +=1
			print("miltn_out"+stra+": ", test(milton_out))
		# print(shake_out)
		# print(twain_out)
		# print(dickens_out)
		# print(doyle_out)
		# print("\n")
	for a in range(19, 30):
		total_tests += 4
		stra = str(a)
		test_shake = prsr.parse(shake+stra+txt)
		test_twain = prsr.parse(twain+stra+txt)
		test_doyle = prsr.parse(doyle+stra+txt)
		test_milton = prsr.parse(milton+stra+txt)
		shake_out = nn.output(test_shake)
		twain_out = nn.output(test_twain)
		doyle_out = nn.output(test_doyle)
		milton_out = nn.output(test_milton)
		# print("test"+stra+":")
		if (test(shake_out) != "shakespeare"): 
			total_wrong += 1
			print("shake_out"+stra + ": ", test(shake_out))
		if (test(twain_out) != "twain"): 
			total_wrong += 1
			print("twain_out"+stra+": ", test(twain_out))
		if (test(doyle_out) != "doyle"): 
			total_wrong += 1
			print("doyle_out"+stra+": ", test(doyle_out))
		if (test(milton_out) != "milton"): 
			total_wrong +=1
			print("miltn_out"+stra+": ", test(milton_out))

	print("correct percentage:", 1 - total_wrong/total_tests)
	print("number wrong: ", total_wrong)
	print("number of tests:", total_tests)
	print("traing rounds:", train_rounds)
	print("learning rate:", learn_rate)


main()