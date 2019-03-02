from neuron import *
from random import uniform
import numpy as np
class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.neurons = []
        for i in range(num_of_neurons):
            weights = np.random.uniform(-1, 1, num_of_inputs)
            bias = np.random.uniform(0,1,1)
            new_neuron = Neuron(weights, bias)
            self.neurons.append(new_neuron)

    def feed_forward(self, inputs):
        output = []
        for neuron in self.neurons:
            output.append(neuron.y(inputs))
        return output
