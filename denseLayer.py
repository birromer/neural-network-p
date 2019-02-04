from math import exp

class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons, base_weight):
        self.num_of_inputs = num_of_inputs
        bw = [base_weight]
        weights = bw * num_of_inputs
        self.neurons = [Neuron(weights, 0) for i in range(len(num_of_neurons))]

    def feed_forward(inputs):
        pass


if __name__ == "__main__":
    pass
