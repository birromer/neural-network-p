from denseLayer import *

class NeuralNetwork:
    def __init__(self, num_of_inputs, neurons_at_layer_1, neurons_at_layer_2):
        self. num_of_inputs = num_of_inputs
        layer_1 = DenseLayer(num_of_inputs, neurons_at_layer_1)
        layer_2 = DenseLayer(neurons_at_layer_1, neurons_at_layer_2)
        
        self.layers = [layer_1, layer_2]
        
    def error_function(self, y, target):
        t = [0 for _ in range(10)]
        t[target] = 1
        sum = 0
        for i in range(len(y)):
            diff = t[i] - y[i]
            sum += (diff**2)
        return sum
        
    def derivative_error_function(self, y, target):
        t = [0 for _ in range(10)]
        t[target] = 1
        error_gradient = []
        for i in range(len(y)):
            partial_d = 2 * (t[i] - y[i])
            #partial_d = -1 * (t[i] - y[i])
            error_gradient.append(partial_d)
        return error_gradient
        
    def feed_forward(self, inputs):
        output_1 = []
        for neuron in self.layers[0].neurons:
            output_1.append(neuron.y(inputs))
        output_2 = []
        for neuron in self.layers[1].neurons:
            output_2.append(neuron.y(output_1))
        return output_2
    
    def one_hot_encoding(x):
        ohe = [0 for _ in range(x)]
        ohe[x] = 1
        return ohe
    
    def encoding_to_number(vec):
        return vec.index(max(vec))
    

    def backpropagation(self, inputs, labels, output, learning_rate):
        dEdy = network.derivative_error_function(output, label)
        
        for i in range(len(network.layers)-1, -1, -1):
            for j in range(len(network.layers[i].neurons)):
                network.layers[i].neurons[j].compute_dEdz(sample, dEdy)
        
        for i in range(len(network.layers)):
            network.layers[i].update_weights(0.001)


if __name__ == "__main__":
    pass

