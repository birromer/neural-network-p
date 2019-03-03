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
            #partial_d = 2 * (t[i] - y[i])
            partial_d = -1 * (t[i] - y[i])
            error_gradient.append(partial_d)
        return error_gradient
        
    def feed_forward(self, inputs):
        outputs = [[] for _ in range(len(self.layers))]
        
        for i in range(len(self.layers)):
            for neuron in self.layers[i].neurons:
                if i != 0:
                    outputs[i] = np.append(outputs[i], neuron.y(outputs[i-1]))
                else:
                    outputs[i] = np.append(outputs[i], neuron.y(inputs))
        return np.array(outputs[-1])
    
    def one_hot_encoding(x):
        ohe = [0 for _ in range(x)]
        ohe[x] = 1
        return ohe
    
    def ohe_to_num(self, vec):
        return vec.argmax(axis=0)
    
    def backpropagation(self, inputs, labels, output, learning_rate):
        dEdy = network.derivative_error_function(output, label)
            
        for i in range(len(self.layers)-1, 0, -1):
            for j in range(len(self.layers[i].neurons)):
                dEdz = self.layers[i].neurons[j].compute_dEdz(dEdy[j]) # CADA DIMENSAO É PARA UM NEURONIOOOOO
                previous_layer_dEdy = np.zeros(len(self.layers[i].neurons[j].weights))
                for k in range(len(self.layers[i].neurons[j].weights)):
                    dEdw = self.layers[i-1].neurons[k].output * dEdz
                    previous_layer_dEdy[k] += dEdz * self.layers[i].neurons[j].weights[k]
                    self.layers[i].neurons[j].update_weights(learning_rate, dEdw)
                self.layers[i].neurons[j].update_bias(learning_rate, dEdz)
            dEdy = np.copy(previous_layer_dEdy)
        
        for j in range(len(network.layers[0].neurons)):
            dEdz = self.layers[0].neurons[j].compute_dEdz(dEdy[j])
            for k in range(len(self.layers[0].neurons[j].weights)):
                dEdw = inputs[k] * dEdz
                self.layers[0].neurons[j].update_weights(learning_rate, dEdw)
            self.layers[i].neurons[j].update_bias(learning_rate, dEdz)



if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    samples = x_train
    labels = y_train

    network = NeuralNetwork(x_train[0].shape[0]**2, [50, 30, 10])

    print("Number of samples used =", len(samples))
    hits = 0
    previous_loss = 0

for p in range(10):
        for s in range(1,len(samples)):
            sample = samples[s].flatten()
            label = labels[s]

            output = network.feed_forward(sample)
            loss = network.error_function(output, label)
            loss_diff = loss - previous_loss
            previous_loss = loss

            output_value = network.ohe_to_num(output)

            if output_value == label:
                hits+=1
            hit_rate = hits/s * 100

            print("Sample %d, label = %d, output = %d" % (s, label, output_value))

            print("Current loss = %.10f, %.2f%% hit rate." %  (loss, hit_rate))

            print("Difference from previous loss = ", loss_diff)

            print(output)

            if s % 5000 == 0 or s % 59999 == 0:
                valids = 0
                for v in range(len(x_test)):
                    o = network.feed_forward(x_test[v].flatten())
                    o = network.ohe_to_num(o)
                    if o == y_test[v]:
                        valids += 1
                hr = valids / len(y_test) * 100
                print("--------------- %d samples analyzed, %f%% hit rate" % (s, hr))
            #    input("press anything to continue")

            network.backpropagation(sample, label, output, 0.001)
        input("press anything to continue")
