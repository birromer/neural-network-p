from math import exp

class Neuron:
    def __init__(self, weights): #add bias later
        self.weights = weights
        #self.bias = bias

    def activation_function(self, input): #sigmoid function
        return 1.0 / (1.0 + exp(-(input)))

    def z(self, inputs): #output
        output = 0
        for i in range(len(inputs)):
            output += (inputs[i] * self.weights[i])
        #print(output)
        return output # + self.bias

    def y(self, inputs):
        #print(self.z(self.y(inputs)))
        return self.activation_function(self.z(inputs))
    
#    def updateWeights(learningRate, dEdy):
#        for i in len(self.weights):
#            weights[i] += learningRate * weight[i] * (target)