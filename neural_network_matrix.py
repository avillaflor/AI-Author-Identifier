import numpy as  np
import math
import pickle
class neural_layer:
	def __init__(self,num_input, num_output, dummy_variable):
		self.theta = np.random.rand(num_output, num_input) * 2 - 1
		self.last_return = None
		self.has_dummy = dummy_variable
		self.update_total = np.zeros((num_output,1))
		self.num_updates = 0

	def add_dummy_output(self, output):
		if (self.has_dummy == 1):
			new_output = np.array([np.append(output, [1])]).T
			return new_output
		return output

	def z(self, input):
		return self.theta.dot(input)

	def output(self, input):
		self.last_input = input
		zout = self.z(input)
		self.last_return = sigmoid(zout)
		return self.add_dummy_output(self.last_return)

	def update(self, prev_delta, learn_coeff):
		if (self.has_dummy):
			prev_delta = prev_delta.copy()[0:-1]
		delta = self.find_delta(prev_delta)
		self.theta -= learn_coeff*(prev_delta.dot(self.last_input.T))
		return delta

	def find_delta(self, prev_delta):
		a = self.last_input
		return (self.theta.T).dot(prev_delta) * g_prime(a)

def sigmoid(z):
	return 1/(1+math.exp(-1*z))
sigmoid = np.vectorize(sigmoid)

def g_prime(x):
	return x * (1-x)
g_prime = np.vectorize(g_prime)

class neural_network:
	
	def __init__(self, num_inputs, num_outputs, layers, dummy_variable, learn_coeff):
		self.nlayer_list = []
		self.last_return = None
		self.learn_coeff = learn_coeff
		self.dummy_var = dummy_variable
		if (len(layers) > 0):
			self.nlayer_list.append(neural_layer(num_inputs , layers[0], dummy_variable))
			for i in range(1,len(layers)):
				self.nlayer_list.append(neural_layer(layers[i-1] + dummy_variable, layers[i], dummy_variable))
			self.nlayer_list.append(neural_layer(layers[-1] + dummy_variable, num_outputs, 0))
		else:
			self.nlayer_list.append(neural_layer(num_inputs, num_outputs, 0))
		
	def addDummyInput(self,input):
		if (self.dummy_var == 1):
			new_input = np.array([np.append(input, [1])]).T
			return new_input
		return input

	def output(self, input):
		o = self.nlayer_list[0].output(input)
		for i in range(1, len(self.nlayer_list)):
			o = self.nlayer_list[i].output(o)
		self.last_return = o
		return o
	
	def back_prop(self, desired_output):
		actual_output = self.last_return
		delta = actual_output - desired_output
		delta = self.nlayer_list[-1].update(delta, self.learn_coeff)
		for i in range (2, len(self.nlayer_list) + 1):
			x = len(self.nlayer_list) - i	
			delta = self.nlayer_list[x].update(delta, self.learn_coeff)

	def out_prop(self, input, desired_output):
		out = self.output(input)
		self.back_prop(desired_output)
		return out

	def update(self):
		for nlayer in self.nlayer_list:
			nlayer.apply_update()

	def save(self, filename):
		pickle.dump(self, open(filename, "wb"))

	def load(filename):
		return pickle.load(open(filename, "rb"))


def main():
	i = np.array([[1.74], [0.4], [0.28]])
	i2 = np.array([[1.93], [.36], [.62]])
	desired = np.array([[.9], [0.1]])
	desired2 = np.array([[0.1], [.9]])

	nn = neural_network(3,2,[5],1, 1)
	for x in range(0,1000):
		if (x%2 == 0):
			nn.output(i)
			nn.back_prop(desired)
		else:
			nn.output(i2)
			nn.back_prop(desired2)
	nn.save("test_network.p")
	nn_copy = neural_network.load("test_network.p")
	print(nn.output(i))
	print(nn.output(i2))

	print(nn_copy.output(i))
	print(nn_copy.output(i2))
# main()