from neuron import *
from random import uniform

def gen_rand_weights(size_vec):
    weights = []
    for _ in range(size_vec):
        weights.append(uniform(-1,1))
    return weights

class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.num_of_inputs = num_of_inputs
        self.neurons = []
        for i in range(num_of_neurons):
            weights = gen_rand_weights(num_of_inputs)
            bias = uniform(0,1)
            new_neuron = Neuron(weights, bias)
            self.neurons.append(new_neuron)

    def feed_forward(self, inputs):
        output = []
        #print("number of neurons ", len(self.neurons))
        for neuron in self.neurons:
            output.append(neuron.y(inputs))
        return output
