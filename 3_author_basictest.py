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

def main():
	shake = "docs2/shakespeare/"
	twain = "docs2/twain/"
	dickens = "docs2/dickens/"
	txt = ".txt"
	nn = nn_matrix.neural_network(6,3,[8],1, .3)
	prsr = ip.parser()
	shake_desired = np.array([[1], [0], [0]])
	twain_desired = np.array([[0], [1], [0]])
	dickens_desired = np.array([[0], [0], [1]])
	input_shake = []
	input_twain = []
	input_dickens = []
	for j in range(1, 15):
		input_shake.append(np.array([prsr.parse(shake+str(j)+txt)]).T)
		input_twain.append(np.array([prsr.parse(twain+str(j)+txt)]).T)
		input_dickens.append(np.array([prsr.parse(dickens+str(j)+txt)]).T)

	# print (input_shake)
	# print(input_twain)
	# print(input_dickens)
	for x in range(0, 100):
		for i in range(0, 14):
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
	for a in range(14,19):
		stra = str(a)
		test_shake = prsr.parse(shake+stra+txt)
		test_twain = prsr.parse(twain+stra+txt)
		test_dickens = prsr.parse(dickens+stra+txt)
		shake_out = nn.output(test_shake)
		twain_out = nn.output(test_twain)
		dickens_out = nn.output(test_dickens)
		print("shake_out"+stra + ": ", test(shake_out))
		print("twain_out"+stra+": ", test(twain_out))
		print("dickens_out"+stra+": ", test(dickens_out))
		print("\n")
		print(shake_out)
		print(twain_out)
		print(dickens_out)
		print("\n")



main()