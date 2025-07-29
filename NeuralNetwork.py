#importing librarys
import numpy as np
from random import *

class Network:
    def __init__(self, inp_size, out_size, hidden_amount, hidden_size):
        self.layers = []
        #instantiate the input layer
        self.layers.append(Layer(0, inp_size, self.layers, 0))
        #instantiate the hidden layers
        for i in range(hidden_amount):
            if i == 0:
                self.layers.append(Layer(inp_size, hidden_size, self.layers, 1))
            else:
                self.layers.append(Layer(hidden_size, hidden_size, self.layers, 2))
        #instantiate the output lauer
        self.layers.append(Layer(hidden_size, out_size, self.layers, 3))
        self.y = self.layers[-1].activation
        for i in range(len(self.layers)):
            self.layers[i].start(i)

    def forward(self):
        #calling forward pass on the last layer
        self.layers[1].forward()
        #return the output layer and the cost of the output layer
        return self.layers[-1], self.layers[-1].cost_calculation(self.y[0])
    
    def backwards(self):
        #call the back pass on the first layer
        self.layers[-1].backwards(self.layers)

class Layer:
    def __init__(self, prev_neuron_amount, curr_neuron_amount, network, current_layer):
        #instantiate the weights matrix with correct dims
        self.weights = np.full((curr_neuron_amount, prev_neuron_amount), 20000)
        #assign the network class so that it can be used in future algorithms
        self.network = network
        #instantiate cost as a float that will be overriden
        self.cost = float(1000)
        #instantiate the activation matrix
        self.activation = np.full((1, curr_neuron_amount), 80000)
        #instantiate the bias matrix
        self.bias = np.full((curr_neuron_amount, 1), 50000)
        #assign the expected ouput y to the output layer and assinging the others zeroes in the correct dims
        if current_layer == len(self.network) - 1:
            self.y = self.network[-1].activation
        else:
            self.y = np.full((curr_neuron_amount, 1), 10000)

    def start(self, current_layer):
        if current_layer != 0:
            self.prev_layer = self.network[current_layer - 1]
        else:
            self.prev_layer = None
        if current_layer != 3:
            self.next_layer = self.network[current_layer + 1]
        else:
            self.next_layer = None

    def forward(self):
        #doing the weighted sum on all of the layers untill all activations are filled
        if self.prev_layer != None:
            print("doing the calculation")
            x = np.matmul(self.weights, self.prev_layer.activation)
            self.activation = x + self.bias
            print(self.activation)
            print("finished the calculation")
        print(self.next_layer)
        if self.next_layer != None:
            self.next_layer.forward()
                
    def cost_calculation(self, expected_output):
        #calculates the cost of a layer given its activation and expected activation
        self.cost = 0
        for i in range(np.shape(self.activation)[0]):
            self.cost += (self.activation[i][0] - self.y[i][0])**2
        return self.cost

    def cost_calculation_dir(self, expected_output):
        #the dirivitive of the cost function for newtons method
        temp = float(0)
        for i in range(np.shape(self.activation)[0]):
            temp += (self.activation[i][0] - self.y[i][0]) * 2
        return temp
    
    def backwards(self, network):
        #calculate the cost of the forward pass
        self.cost = self.cost_calculation(self.y)
        print(self.cost)

        #changing weights
        print(self.weights)
        for j in range(np.shape(self.weights)[0]):
            for i in range(np.shape(self.weights)[1]):
                #newtons method
                print("start of NM for weights")
                if ((self.prev_layer.activation[j][0]) * SIG_DIR(self.activation, j) * self.cost_calculation_dir(self.y[j][0])) != 0:
                    self.weights[j][i] -= ((self.cost)/((self.prev_layer.activation[j][0]) * SIG_DIR(self.activation, j) * self.cost_calculation_dir(self.y[j][0])))
                else:
                    self.weights[j][i] -= 0.001
                print("end of NM for weights")
        print(self.weights)

        #create y for new layer
        if self.prev_layer != None:
            print("starting to make new Y")
            for i in range(np.shape(self.prev_layer.activation)[0]):
                temp_y = 0
                temp_weights = 0
                for j in range(np.shape(self.activation)[1]):
                    #get temporary weights to be used in the later algorithm
                    temp_weights += self.weights[j][i]
                for j in range(np.shape(self.prev_layer.activation)[1]):
                    #newtons method
                    if ((temp_weights * (SIG_DIR(self.prev_layer.activation, j)) * self.cost_calculation_dir(self.y[j]))) != 0:
                        temp_y += self.cost / (temp_weights * (SIG_DIR(self.prev_layer.activation, j)) * self.cost_calculation_dir(self.y[j]))
                    else:
                        temp_y += 0.001
                print(temp_y)
                self.prev_layer.y[i][0] = temp_y
            print("finishing making new Y")

        #changing bias
        print("starting to make new biases")
        for i in range(np.shape(self.weights)[1]):
            for j in range((np.shape(self.bias)[0])):
                #newtons method
                if () != 0:
                    self.weights[j][i] -= self.cost/(SIG_DIR(self.activation, j) * self.cost_calculation_dir(self.y[j]))
                else:
                    self.weights[j][i] -= 0.001
        print("finished changing biases")
    
        if self.prev_layer != None:
            #do the back pass on the previous layer if it is not the input layer
            self.prev_layer.backwards(self.network)

def SIG(arr):
    #sigmoid activation function
    for i in range(np.shape(arr)[0]):
        #arr[i][0] = 1/(1 + np.exp(-arr[i][0]))
    #return arr
        if arr[i][0] < 0:
            arr[i][0] = 0
    return arr

def SIG_DIR(val, i):
    #dirivitive of the sigmoid activation function for newtons method
    return (SIG(val)[i][0] * (1- SIG(val)[i][0]))