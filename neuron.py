from math import exp
    
class Neuron:
    def __init__(self, weights, bias=0): # add bias later
        self.weights = weights
        self.bias = bias
        self.dEdz = 0
        self.output = 0

    def activation_function(self, input): # sigmoid function
        #print("input ",input)
        return 1.0 / (1.0 + exp(-(input)))

    def z(self, inputs): # output
        output = 0
        for i in range(len(inputs)):
            output += (inputs[i] * self.weights[i])
        return output # + self.bias

    def y(self, inputs):
        self.output = self.activation_function(self.z(inputs))
        return self.output
    
    def compute_dEdz(self, dEdy): 
        y = self.output
        self.dEdz =  (y * (1 - y) * dEdy)
        return self.dEdz
    
    def update_weights(self, learningRate, dEdw):
        for i in range(len(self.weights)):
            self.weights[i] -= learningRate * dEdw
            
    def update_bias(self, learning_rate, dEdz):
        self.bias -= learning_rate * dEdz

if __name__ == "__main__":
    inputs = [2, 5, 3]

    neuron = Neuron([150, 50, 100])

    print(neuron.z(inputs))