import neural_network_matrix as nn_matrix
import info_parser as ip
import numpy as np

def main():
	shake = "docs/shakespeare-"
	txt = ".txt"
	twain = "docs/twain-"
	nn = nn_matrix.neural_network(6,2,[8],1, 0.5)
	prsr = ip.parser()
	shake_desired = np.array([[1], [0]])
	twain_desired = np.array([[0], [1]])
	input_shake = []
	input_twain = []
	for j in range(1, 6):
		input_shake.append(np.array([prsr.parse(shake+str(j)+txt)]).T)
		input_twain.append(np.array([prsr.parse(twain+str(j)+txt)]).T)

	print (input_shake)
	print(input_twain)
	for x in range(0, 1000):
		for i in range(0, 5):
			# input_shake = np.array([prsr.parse(shake+str(i)+txt)]).T
			# print("input_shake: ", input_shake)
			# input_twain = np.array([prsr.parse(twain+str(i)+txt)]).T
			# print("input_twain: ", input_twain)
			nn.output(input_shake[i])
			nn.back_prop(shake_desired)
			nn.output(input_twain[i])
			nn.back_prop(twain_desired)
	test_shake = prsr.parse(shake+"test1"+txt)
	test_twain = prsr.parse(twain+"test1"+txt)
	shake_out = nn.output(test_shake)
	twain_out = nn.output(test_twain)
	print("shake_out: ", shake_out)
	print("twain_out: ", twain_out)
main()