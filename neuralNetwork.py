from denseLayer import *

class NeuralNetwork:
    def __init__(self, num_of_inputs, neurons_at_layer_1, neurons_at_layer_2):
        self. num_of_inputs = num_of_inputs
        layer_1 = DenseLayer(num_of_inputs, neurons_at_layer_1)
        layer_2 = DenseLayer(neurons_at_layer_1, neurons_at_layer_2)
        self.layers = [layer_1, layer_2]

    def feed_forward(self, inputs):
        output_1 = []
        for neuron in self.layers[0].neurons:
            output_1.append(neuron.y(inputs))
        output_2 = []
        for neuron in self.layers[1].neurons:
            output_2.append(neuron.y(output_1))
        return output_2


if __name__ == "__main__":
    pass

