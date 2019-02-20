from neuron import *
from random import uniform

def gen_rand_weights(size_vec):
    weights = []
    for _ in range(size_vec):
        weights.append(uniform(0,0.5))
    return weights

class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.num_of_inputs = num_of_inputs
        self.neurons = []
        for i in range(num_of_neurons):
            weights = gen_rand_weights(num_of_inputs)
            new_neuron = Neuron(weights)
            self.neurons.append(new_neuron)

    def feed_forward(self, inputs):
        output = []
        for neuron in self.neurons:
            output.append(neuron.y(inputs))
            #print(neuron.activation_function(inputs))
        return output
    
    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)

if __name__ == "__main__":
    pass
