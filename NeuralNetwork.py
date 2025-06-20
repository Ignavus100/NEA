#importing librarys
import numpy as np
from random import *

class Network:
    def __init__(self, inp_size, out_size, hidden_amount, hidden_size):
        self.layers = []
        y = np.matrix(1)#TODO: from traning file
        #instantiate the input layer
        self.layers.append(Layer(0, inp_size, self.layers, 0))
        #instantiate the hidden layers
        for i in range(hidden_amount):
            if i == 0:
                self.layers.append(Layer(inp_size, hidden_size, self.layers, i + 1))
            else:
                self.layers.append(Layer(hidden_size, hidden_size, self.layers, i + 1))
        #instantiate the output lauer
        self.layers.append(Layer(hidden_size, out_size, self.layers, i + 2))

    def forward(self):
        #calling forward pass on the last layer
        self.layers[-1].forward()
        #return the output layer and the cost of the output layer
        return self.layers[-1], self.layers[-1].cost(self.y)
    
    def backwards(self):
        #call the back pass on the first layer
        self.layers[0].backwards()

class Layer:
    def __init__(self, prev_neuron_amount, curr_neuron_amount, network, current_layer):
        #instantiate the weights matrix with correct dims
        self.weights = np.full((curr_neuron_amount, prev_neuron_amount), 2)
        #assign the network class so that it can be used in future algorithms
        self.network = network
        #instantiate cost as a float that will be overriden
        self.cost = float(0)
        #assign the previous layer if the current layer is not the input layer
        if current_layer != 0:
            self.prev_layer = network[current_layer - 1]
        else:
            self.prev_layer = None
        #instantiate the activation matrix
        self.activation = np.full((curr_neuron_amount, 1), 8)
        #instantiate the bias matrix
        self.bias = np.full((prev_neuron_amount, 1), 5)
        #assign the expected ouput y to the output layer and assinging the others zeroes in the correct dims
        if current_layer == len(self.network) - 1:
            y = self.network[-1]
        else:
            y = np.full((curr_neuron_amount, 1), 0)

    def forward(self):
        #doing the weighted sum on all of the layers untill all activations are filled
        while self.activation[0] != None:
            if self.prev_layer != None:
                self.activation = SIG(self.weights * self.prev_layer.activation + self.bias)
            else:
                if self.prev_layer != self.network[0]:
                    self.prev_layer.forward()
                
    def cost_calculation(self, expected_output):
        #calculates the cost of a layer given its activation and expected activation
        for i in range(len(expected_output)):
            self.cost += (self.activation[i] - expected_output[i])**2

    def cost_calculation_dir(self, expected_output):
        #the dirivitive of the cost function for newtons method
        temp = float(0)
        for i in range(len(expected_output)):
            temp += (self.activation[i] - expected_output[i]) * 2
        return temp
    
    def backwards(self, network):
        #filling the network with activations on the forward pass
        network.forward()
        #calculate the cost of the forward pass
        self.cost = self.cost_calculation(self.y)

        #changing weights
        for i in range(np.shape(self.weights)[1]):
            for j in range(np.shape(self.weights)[0]):
                #newtons method
                self.weights[i][j] = self.weights[i][j] - (self.cost)/(self.prev_layer.activation[j]) * SIG_DIR(self.weights * self.prev_layer.activation + self.bias)[j] * self.cost_calculation_dir(self.y[j])

        #create y for new layer
        for i in range(np.shape(self.prev_layer.activation)[0]):
            temp_y = 0
            temp_weights = 0
            for j in range(np.shape(self.activation)[0]):
                #get temporary weights to be used in the later algorithm
                temp_weights += self.weights[i][j]
            for j in range(np.shape(self.activation)[0]):
                #newtons method
                temp_y += (temp_weights * (SIG_DIR(self.weights * self.prev_layer.activation + self.bias)[j]) * self.cost_calculation_dir(self.y[j]))
            self.prev_layer.y.append(temp_y)

        #changing bias
        for i in range(np.shape(self.weights)[1]):
            for j in range((np.shape(self.bias)[0])):
                #newtons method
                self.weights[i][j] = self.weights[i][j] - self.cost/(SIG_DIR(self.weights * self.prev_layer.activation + self.bias)[j] * self.cost_calculation_dir(self.y[j]))

            if self.prev_layer != None:
                #do the back pass on the previous layer if it is not the input layer
                self.prev_layer.backwards()

def SIG(arr):
    #sigmoid activation function
    for i in range(len(arr)):
        arr[0][i] = 1/(1 + np.exp(-arr[0][i]))
    return arr

def SIG_DIR(val):
    #dirivitive of the sigmoid activation function for newtons method
    return SIG(val)(1- SIG(val))