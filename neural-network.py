from denseLayer import *

class NeuralNetwork:
    def __init__(self, num_of_inputs, [neurons_at_layer_1, neurons_at_layer_2]):
        self. num_of_inputs = num_of_inputs
        layer_1 = DenseLayer(num_of_inputs, neurons_at_layer_1)
        layer_2 = DenseLayer(neurons_at_layer_1, 10)    # 10 possible outputs in the one hot encoding
        self.layers = [layer_1, layer_2]


