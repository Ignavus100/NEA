import numpy as np
from random import *

class Network:
    def __init__(self, inp_size, out_size, hidden_amount, hidden_size):
        layers = []
        y = np.matrix()#from traning data
        layers.append(Layer(0, inp_size, self.layers, 0))
        for i in range(hidden_amount):
            if i == 0:
                layers.append(Layer(inp_size, hidden_size, self.layers, i + 1))
            else:
                layers.append(Layer(hidden_size, hidden_size, self.layers, i + 1))
        layers.append(Layer(hidden_size, out_size, self.layers, i + 2))

    def forward(self):
        self.layers[-1].forward()
        return self.layers[-1], self.layers[-1].cost(self.y)
    
    def backwards(self):
        self.layers[0].backwards()

class Layer:
    def __init__(self, prev_neuron_amount, curr_neuron_amount, network, current_layer):
        self.weights = np.full((curr_neuron_amount, prev_neuron_amount), 2)
        self.network = network
        self.cost = float(0)
        if current_layer != 0:
            self.prev_layer = network[current_layer - 1]
        else:
            self.prev_layer = None
        self.activation = np.full((curr_neuron_amount, 1), 8)
        self.bias = np.full((prev_neuron_amount, 1), 5)
        if current_layer == len(network.layers) - 1:
            y = network.y
        else:
            y = np.full((curr_neuron_amount, 1), 0)

    def forward(self):
        while self.activation[0] != None:
            if self.prev_layer != None:
                self.activation = SIG(self.weights * self.prev_layer.activation + self.bias)
            else:
                if self.prev_layer != self.network[0]:
                    self.prev_layer.forward()
                
    def cost_calculation(self, expected_output):
        for i in range(len(expected_output)):
            self.cost += (self.activation[i] - expected_output[i])**2

    def cost_calculation_dir(self, expected_output):
        temp = float(0)
        for i in range(len(expected_output)):
            temp += (self.activation[i] - expected_output[i]) * 2
        return temp
    
    def backwards(self, network):
        network.forward()
        self.cost = self.cost_calculation(self.y)

        #changing weights
        for i in range(np.shape(self.weights)[1]):
            for j in range(np.shape(self.weights)[0]):
                self.weights[i][j] = self.weights[i][j] - (self.cost)/(self.prev_layer.activation[j]) * SIG_DIR(self.weights * self.prev_layer.activation + self.bias)[j] * self.cost_calculation_dir(self.y[j])

        #create y for new layer
        for i in range(np.shape(self.prev_layer.activation)[0]):
            temp_y = 0
            temp_weights = 0
            for j in range(np.shape(self.activation)[0]):
                temp_weights += self.weights[i][j]
            for j in range(np.shape(self.activation)[0]):
                temp_y += (temp_weights * (SIG_DIR(self.weights * self.prev_layer.activation + self.bias)[j]) * self.cost_calculation_dir(self.y[j]))
            self.prev_layer.y.append(temp_y)

        #changing bias
        for i in range(np.shape(self.weights)[1]):
            for j in range((np.shape(self.bias)[0])):
                self.weights[i][j] = self.weights[i][j] - self.cost/(SIG_DIR(self.weights * self.prev_layer.activation + self.bias)[j] * self.cost_calculation_dir(self.y[j]))

            if self.prev_layer != None:
                self.prev_layer.backwards

def SIG(arr):
    for i in range(len(arr)):
        arr[0][i] = 1/(1 + np.exp(-arr[0][i]))
    return arr

def SIG_DIR(val):
    return SIG(val)(1- SIG(val))