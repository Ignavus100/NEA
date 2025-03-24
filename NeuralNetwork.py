import numpy as np
import random
from DatabaseAccess import select
import pickle

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
        self.prevLayer = prevLayer
        temp = 0
        activation = Activation_ReLU()
        for i in range(len(self.prevLayer)):
            if self.weights != None and self.prevLayer[i].val != None and self.bias != None:
                #print(self.weights[i], self.bias, self.prevLayer[i].val)
                activation.forward((self.bias + (float(self.weights[i]) * float(self.prevLayer[i].val))))
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

    def newInput(self, values):
        for i in range(self.inp_size):
            self.layers[0][i] = (Node(None, values[i]))

    def forward(self):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].forward(self.layers[i-1])

        temp = []
        for i in range(len(self.layers[-1])):
            temp.append(self.layers[-1][i].val)
        
        activation = Activation_Softmax()
        activation.forward(temp)
        return activation.output
    
    def backwards(self):
        #weights new calculation
        #TODO: make it so that you can get the new biases by doing newtons method (on paper) and using charin rule for the derrivitave of the cost with respect to the weight
        cost = cost_calculation()
        for i in range((len(self.layers) - 2), 0, -1):
            for j in self.layers[i]:
                for k in range(len(j.weights)):
                    if i == len(self.layers):
                        z = j.bias + (float(j.weights[k]) * float(j.prevLayer[i].val))
                        if z < 0:
                            mz = 0
                        else:
                            mz = 1
                            if (float(j.prevLayer[i].val) * mz * (2*(j.val - y[k]))) != 0:
                                j.weights[k] -= ((cost.calculate(n.forward(), y))/(float(j.prevLayer[i].val) * mz * (2*(j.val - y[k]))))

                    else:
                        z = j.bias + (j.weights[k] * float(j.prevLayer[i].val))
                        if z < 0:
                            mz = 0
                        else:
                            mz = 1
                            if (j.weights[k] * mz * (2*(j.val))) != 0:
                                j.weights[k] -= ((cost.calculate(n.forward(), y))/(j.weights[k] * mz * (2*(j.val))))


        #TODO: do the same but instead of weights, biases
        for i in range(len(self.layers) - 2, 0, -1):
            for j in self.layers[i]:
                for k in range(len(j.weights)):
                    if i == len(self.layers):
                        z = j.bias + (j.weights[k] * float(j.prevLayer[i].val))
                        if z < 0:
                            mz = 0
                        else:
                            mz = 1
                            if (mz * (2*(j.val - y[k]))) != 0:
                                j.weights[k] -= ((cost.calculate(n.forward(), y))/(mz * (2*(j.val - y[k]))))

                    else:
                        z = j.bias + (j.weights[k] * float(j.prevLayer[i].val))
                        if z < 0:
                            mz = 0
                        else:
                            mz = 1
                            if (mz * (2*(j.val))) !=  0:
                                j.bias -= ((cost.calculate(n.forward(), y))/(mz * (2*(j.val))))



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
        temp = []
        for j in range(len(y_pred)):
            temp.append(0)
        temp[y_true[0]] = 1
        y_resolved.append(temp)
        for i in range(samples):
            cost += (y_pred[i] - y_resolved[0][i])**2
        return cost


def normalizeData(data):
    normalized_data = []
    sigma = 0
    for i in data:
        sigma += i
    for i in data:
        normalized_data.append(i/sigma)
    return normalized_data


batch_length = int(input("batch length: "))
flattened_X = []
for j in range(batch_length):
    X = []
    for i in range(15):
        X.append(select("c, h, l, o, v", "AAPL", f"ID = {15 * j + i}"))
    flattened_X.append([value for sublist in X for inner_list in sublist for value in inner_list])#flattened, is one dimensional

y = []#expected outcomes
highNext15 = []
for j in range(batch_length):
    for i in range(15):
        try:
            if highNext15[j] < select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + 15 + i}")[0][0]:
                highNext15[j] = select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + 15 + i}")[0][0]
        except:
            highNext15.append(select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + 15 + i}")[0][0])

high15 = []
for j in range(batch_length):
    for i in range(15):
        try:
            if high15[j] < select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + i}")[0][0]:
                high15[j] = select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + i}")[0][0]
        except:
            high15.append(select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + i}")[0][0])

for i in range(batch_length):
    if highNext15[i] > high15[i]:
        val = 1
    elif highNext15[i] < high15[i]:
        val = 0
    else:
        try:
            val = y[i][-1]
        except:
            val = 0
    y.append(val)

flattened_X = normalizeData(flattened_X[0])

def save_model(network, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(network, f)

def load_model(filename="model.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

n = load_model()

def testing(epochs):
    for epoch in range(epochs):
        for i in range(2000):
            flattened_X = []
            for j in range(batch_length):
                X = []
                for k in range(15):
                    X.append(select("c, h, l, o, v", "AAPL", f"ID = {15*(i+1) + k}"))
                #flattened_X.append([value for sublist in X for inner_list in sublist for value in inner_list])#flattened, is one dimensional
            #print(X)
            flattened_X.append([value for sublist in X for inner_list in sublist for value in inner_list])#flattened, is one dimensional
            y = []#expected outcomes
            highNext15 = []
            for j in range(batch_length):
                for i in range(15):
                    try:
                        if highNext15[j] < select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + 15 + i}")[0][0]:
                            highNext15[j] = select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + 15 + i}")[0][0]
                    except:
                        highNext15.append(select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + 15 + i}")[0][0])

            high15 = []
            for j in range(batch_length):
                for i in range(15):
                    try:
                        if high15[j] < select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + i}")[0][0]:
                            high15[j] = select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + i}")[0][0]
                    except:
                        high15.append(select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + i}")[0][0])

            for i in range(batch_length):
                if highNext15[i] > high15[i]:
                    val = 1
                elif highNext15[i] < high15[i]:
                    val = 0
                else:
                    #try:
                    val = y[i][-1]
                    #except:
                        #val = 0
                y.append(val)

            
            flattened_X = normalizeData(flattened_X[0])
            n.newInput(flattened_X)

            cost = cost_calculation()
            print(y)
            temp = n.forward()
            print(temp)
            print(cost.calculate(temp, y))
            n.backwards()

            save_model(n)

epochs = int(input("epochs: "))
testing(epochs)