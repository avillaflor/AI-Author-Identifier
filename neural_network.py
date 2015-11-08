from random import random
from math import exp
class neuron:
	def __init__(self, num_weights):
		self.num_weights = num_weights
		self.weights = []
		for x in range (0, num_weights):
			random_num = random()*2 - 1
			# print(random_num)
			self.weights.append(random_num)

	def z(self, inputs):
		total = 0
		for x in range(0, self.num_weights):
			total += inputs[x] * self.weights[x]
		return total
	def output(self, inputs):
		z_out = self.z(inputs);
		y = 1/(1+exp(-1*z_out))
		return y
class const_neuron:
	def __init__(self, const = 1):
		self.const = const
	def output(self, input):
		return self.const
class neural_layer:
	def __init__(self, num_neurons, num_inputs, num_dummy = 0):
		self.num_neurons = num_neurons
		self.num_inputs = num_inputs
		self.neuron_list = []
		for x in range(0,num_neurons):
			self.neuron_list.append(neuron(num_inputs))
		for x in range(0,num_dummy):
			self.neuron_list.append(const_neuron())
	def output(self, inputs):
		output_vector = []
		for x in range(0, len(self.neuron_list)):
			output_vector.append(self.neuron_list[x].output(inputs))
		return output_vector
class neural_map:
	def __init__(self, n_layer, num_inputs, dummy_per_layer = 0):
		self.nlayer_list = []
		self.num_inputs = num_inputs
		for x in range(0, len(n_layer)):
			if (x == 0):
				self.nlayer_list.append(neural_layer(n_layer[0], num_inputs, dummy_per_layer))
			else:
				self.nlayer_list.append(neural_layer(n_layer[x], n_layer[x-1]))
		self.nlayer_list.append(neuron(n_layer[len(n_layer)-1]))
	def output(self, inputs):
		o = inputs
		for x in range(0,len(self.nlayer_list)-1):
			o = self.nlayer_list[x].output(o)
		return self.nlayer_list[len(self.nlayer_list)-1].z(o)

def main():
	nm = neural_map([100,100,100], 4, 1)
	out = nm.output([1,1,1,1])
	print(out)



main()