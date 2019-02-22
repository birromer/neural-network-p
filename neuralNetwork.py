from denseLayer import *

class NeuralNetwork:
    def __init__(self, num_of_inputs, layers):
        self. num_of_inputs = num_of_inputs
        self.layers = [DenseLayer(num_of_inputs, layers[0])]
        for i in range(1,len(layers)):
            added_layer = DenseLayer(layers[i-1], layers[i])
            self.layers.append(added_layer)
                    
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
    
    def ohe_to_num(self, vec):
        return vec.index(max(vec))
    

    def backpropagation(self, inputs, labels, output, learning_rate):
        dEdy = network.derivative_error_function(output, label)
                
        for i in range(len(self.layers)-1, 0, -1):
            for j in range(len(self.layers[i].neurons)):
                dEdz = self.layers[i].neurons[j].compute_dEdz(dEdy[j]) # CADA DIMENSAO É PARA UM NEURONIOOOOO
                for k in range(len(self.layers[i-1].neurons)):
                    dEdw = self.layers[i-1].neurons[k].output * dEdz
                    weights = self.layers[i-1].neurons[k].weights
                    previous_layer_dEdy = [weight * dEdz for weight in weights]            
                    self.layers[i].neurons[j].update_weights(learning_rate, dEdw)
        
        dEdy = previous_layer_dEdy.copy()
        
        for j in range(len(network.layers[0].neurons)):
            dEdz = self.layers[0].neurons[j].compute_dEdz(dEdy[j])
            for k in range(len(self.layers[0].neurons[j].weights)):
                dEdw = inputs[k] * dEdz
                self.layers[0].neurons[j].update_weights(learning_rate, dEdw)
if __name__ == "__main__":
    pass

