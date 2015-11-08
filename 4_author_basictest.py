import neural_network_matrix as nn_matrix
import info_parser as ip
import numpy as np

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

def main():
	shake = "docs2/shakespeare/"
	twain = "docs2/twain/"
	dickens = "docs2/dickens/"
	doyle = "docs2/doyle/"
	txt = ".txt"
	nn = nn_matrix.neural_network(7,4,[20,20],1, .25)
	prsr = ip.parser()
	shake_desired = np.array([[1], [0], [0], [0]])
	twain_desired = np.array([[0], [1], [0], [0]])
	dickens_desired = np.array([[0], [0], [1], [0]])
	doyle_desired = np.array([[0], [0], [0], [1]])
	input_shake = []
	input_twain = []
	input_dickens = []
	input_doyle = []
	for j in range(1, 16):
		input_shake.append(np.array([prsr.parse(shake+str(j)+txt)]).T)
		input_twain.append(np.array([prsr.parse(twain+str(j)+txt)]).T)
		input_dickens.append(np.array([prsr.parse(dickens+str(j)+txt)]).T)
		input_doyle.append(np.array([prsr.parse(doyle+str(j)+txt)]).T)

	# print (input_shake)
	# print(input_twain)
	# print(input_dickens)
	for x in range(0, 1000):
		for i in range(0, 15):
			# input_shake = np.array([prsr.parse(shake+str(i)+txt)]).T
			# print("input_shake: ", input_shake)
			# input_twain = np.array([prsr.parse(twain+str(i)+txt)]).T
			# print("input_twain: ", input_twain)
			nn.output(input_shake[i])
			nn.back_prop(shake_desired)
			nn.output(input_twain[i])
			nn.back_prop(twain_desired)
			nn.output(input_dickens[i])
			nn.back_prop(dickens_desired)
			nn.output(input_doyle[i])
			nn.back_prop(doyle_desired)
	total_wrong = 0
	total_tests = 0
	for a in range(15,19):
		total_tests += 4
		stra = str(a)
		test_shake = prsr.parse(shake+stra+txt)
		test_twain = prsr.parse(twain+stra+txt)
		test_dickens = prsr.parse(dickens+stra+txt)
		test_doyle = prsr.parse(doyle+stra+txt)
		shake_out = nn.output(test_shake)
		twain_out = nn.output(test_twain)
		dickens_out = nn.output(test_dickens)
		doyle_out = nn.output(test_doyle)
		print("shake_out"+stra + ": ", test(shake_out))
		print("twain_out"+stra+": ", test(twain_out))
		print("dickens_out"+stra+": ", test(dickens_out))
		print("doyle_out"+stra+": ", test(doyle_out))
		print("\n")
		if (test(shake_out) != "shakespeare"): total_wrong += 1
		if (test(twain_out) != "twain"): total_wrong += 1
		if (test(dickens_out) != "dickens"): total_wrong += 1
		if (test(doyle_out) != "doyle"): total_wrong += 1
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
		shake_out = nn.output(test_shake)
		twain_out = nn.output(test_twain)
		doyle_out = nn.output(test_doyle)
		print("shake_out"+stra + ": ", test(shake_out))
		print("twain_out"+stra+": ", test(twain_out))
		print("doyle_out"+stra+": ", test(doyle_out))
		print("\n")
		if (test(shake_out) != "shakespeare"): total_wrong += 1
		if (test(twain_out) != "twain"): total_wrong += 1
		if (test(doyle_out) != "doyle"): total_wrong += 1

	print("correct percentage: ", 1 - total_wrong/total_tests)
	print(total_wrong, total_tests)


main()