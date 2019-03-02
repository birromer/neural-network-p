from math import exp
import numpy as np
class Neuron:
    def __init__(self, weights, bias=0): # add bias later
        self.weights = weights
        self.bias = bias
        self.dEdz = 0
        self.output = 0

    def activation_function(self, input): # sigmoid function
        return 1.0 / (1.0 + np.exp(-(input)))

    def z(self, inputs): # output
        output = inputs.dot(self.weights)
        return output + self.bias

    def y(self, inputs):
        self.output = self.activation_function(self.z(inputs))
        return self.output
    
    def compute_dEdz(self, dEdy): 
        y = self.output
        self.dEdz =  (y * (1 - y) * dEdy)
        return self.dEdz
    
    def update_weights(self, learningRate, dEdw):
        self.weights -= (learningRate * dEdw)
            
    def update_bias(self, learning_rate, dEdz):
        self.bias -= learning_rate * dEdz

if __name__ == "__main__":
    inputs = [2, 5, 3]

    neuron = Neuron([150, 50, 100])

    print(neuron.z(inputs))