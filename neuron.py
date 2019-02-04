from math import exp

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        # self.bias = bias

    def z(self, input): #sigmoid function
        return (1 / 1 + exp(-input))

    def y(self, inputs): #output
        output = 0
        for i in range(len(inputs)):
            output += (inputs[i] * self.weights[i])
        return output # + self.bias

    def activation_function(self, inputs):
        return sigmoid(output(inputs))

#    def updateWeights(learningRate, dEdy):
#        for i in len(self.weights):
#            weights[i] += learningRate * weight[i] * (target)


if __name__ == "__main__":
    inputs = [2, 5, 3]

    neuron = Neuron([150, 50, 100], 0)

    print(neuron.y(inputs))
