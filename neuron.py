from math import exp

class Neuron:
    def __init__(self, weights, bias=0): # add bias later
        self.weights = weights
        self.bias = bias
        self.dEdz = 0

    def activation_function(self, input): # sigmoid function
        return 1.0 / (1.0 + exp(-(input)))

    def z(self, inputs): # output
        output = 0
        for i in range(len(inputs)):
            output += (inputs[i] * self.weights[i])
        return output # + self.bias

    def y(self, inputs):
        return self.activation_function(self.z(inputs))
    
    def compute_dEdz(self, inputs, dEdy): # dEdy is a gradient
        y = self.y(inputs)
        self.dEdz =  (y * (1 - y) * dEdy)
    
    def update_weights(self, learningRate):
        for i in range(len(self.weights)):
            self.weights[i] += learningRate * self.weights[i] * self.dEdz

if __name__ == "__main__":
    inputs = [2, 5, 3]

    neuron = Neuron([150, 50, 100])

    print(neuron.z(inputs))