from math import exp

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        # self.bias = bias

    def sigmoid(input):
        return (1 / 1 + exp(-input))

    def output(inputs):
        output = 0
        for i in len(inputs):
            sum += input[i] * weight[i]
        return output # + self.bias

    def activate(inputs):
        return sigmoid(output(inputs))

    def updateWeights(learningRate, dEdy):
        for i in len(self.weights):
            weights[i] += learningRate * weight[i] * (target -
