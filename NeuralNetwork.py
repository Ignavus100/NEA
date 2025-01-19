import numpy as np
import nnfs
import random
from nnfs.datasets import spiral_data
from DatabaseAccess import select


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        val = np.min(inputs)
        if val < 0:
            for i in range(len(inputs)):
                inputs[i] = inputs[i] - val
        
        probabilities = normalizeData(inputs)
        self.output = probabilities


class Node:
    def __init__(self, prevNeuronsAmount: int, val):
        self.val = val
        self.bias = 0.1 * random.randint(-50, 50)
        self.prevNeuronsAmount = prevNeuronsAmount
        self.weights = []
        if prevNeuronsAmount != None:
            for i in range(prevNeuronsAmount):
                self.weights.append(0.1 * random.randint(-20, 20))
        else:
            self.weights = None


    def forward(self, prevLayer: list):
        temp = 0
        activation = Activation_ReLU()
        for i in range(len(prevLayer)):
            if self.weights != None:
                activation.forward((self.bias + (self.weights[i] * float(prevLayer[i].val))))
            else:
                temp = 0
            temp += activation.output
        self.val = temp



class Network:
    def __init__(self, hidden: int, inp_size: int, out_size: int, values: list):
        self.hidden = hidden
        self.inp_size = inp_size
        self.out_size = out_size
        self.layers = []

        temp = []
        for i in range(inp_size):
            temp.append(Node(None, values[i]))
        self.layers.append(temp)

        temp = []
        for i in range(hidden):
            temp = []
            for j in range(16):
                if i == 0:
                    temp.append(Node(inp_size, None))
                else:
                    temp.append(Node(16, None))
            self.layers.append(temp)

        temp = []
        for i in range(out_size):
            temp.append(Node(16, None))
        self.layers.append(temp)


    def forward(self):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].forward(self.layers[i-1])

        temp = []
        for i in range(len(self.layers[-1])):
            temp.append(self.layers[-1][i].val)
        print(temp)
        
        activation = Activation_Softmax()
        activation.forward(temp)
        return activation.output



class cost:
    def calculate(self, output, y):
        cost = self.forward(output, y)
        #data_loss = np.mean(sample_losses)
        return cost



class cost_calculation(cost):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        cost = 0
        y_resolved = []
        for i in y_true:
            temp = []
            for j in range(len(y_pred[0])):
                temp.append(0)
            temp[i] = 1
            y_resolved.append(temp)
        for i in range(samples):
            for j in range(len(y_pred[0])):
                cost += (y_pred[i][j] - y_resolved[i][j])**2
        return cost


def normalizeData(data):
    '''
    data = np.array(data)
    min_val, max_val = np.min(data), np.max(data)
    range_min, range_max = (0, 1)

    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = normalized_data * (range_max - range_min) + range_min
    '''
    normalized_data = []
    print(data)
    sigma = 0
    for i in data:
        sigma += i
    for i in data:
        normalized_data.append(i/sigma)
    return normalized_data

def backwards(cost):
    pass
#X, y =  spiral_data(samples=100, classes=3)
flattened_X = []
batch_length = int(input("batch length: "))
for j in range(batch_length):
    X = []
    rnum = random.randint(1, 30125)
    for i in range(10):
        X.append(select("*", "AAPL", f"ID = {rnum + i}"))
    flattened_X.append([value for sublist in X for inner_list in sublist for value in inner_list])#flattened is one dimensional

y = []#expected outcomes
for i in flattened_X:
    if select("o", "AAPL", f"ID = {int(i[0]) + 15}")[0][0] > float(i[4]):
        val = 1
    elif select("o", "AAPL", f"ID = {int(i[0]) + 15}")[0][0] < float(i[4]):
        val = 0
    else:
        val = 1
    y.append(val)

flattened_X = normalizeData(flattened_X[0])


n = Network(2, len(flattened_X), 2, flattened_X)
print(n.forward())

































'''
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        inputs = np.array(inputs, dtype=np.float32)
        self.output = np.dot(inputs, self.weights) + self.biases
'''
