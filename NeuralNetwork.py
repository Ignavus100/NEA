import numpy as np
import random
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
        self.bias = np.random.randn() * 0.1
        self.weights = np.random.randn(prevNeuronsAmount) * np.sqrt(2/prevNeuronsAmount) if prevNeuronsAmount else None
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
        cost = cost_calculation()
        loss = cost.calculate(self.forward(), y)
        
        for i in range(len(self.layers) - 1, 0, -1):
            for j in self.layers[i]:
                for k in range(len(j.weights)):
                    prev_val = float(j.prevLayer[k].val) if j.prevLayer is not None else 0
                    
                    z = j.bias + (j.weights[k] * prev_val)
                    mz = 1 if z > 0 else 0  # Derivative of ReLU
                    
                    y_target = y[k] if k < len(y) else 0  # Ensure valid indexing
                    
                    gradient_w = 2 * (j.val - y_target) * mz * prev_val if i == len(self.layers) - 1 else 2 * j.val * mz * prev_val
                    gradient_b = 2 * (j.val - y_target) * mz if i == len(self.layers) - 1 else 2 * j.val * mz
                    
                    if gradient_w != 0:
                        j.weights[k] -= loss / gradient_w
                    
                # Update biases
                if gradient_b != 0:
                    j.bias -= loss / gradient_b


class cost:
    def calculate(self, output, y):
        cost = self.forward(output, y)
        #data_loss = np.mean(sample_losses)
        return cost


class cost_calculation(cost):
    def forward(self, y_pred, y_true):
        # Convert y_true to numpy array first
        y_true = np.array(y_true)
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            y_true = np.eye(2)[y_true.astype(int)]
        return -np.mean(y_true * np.log(y_pred))

    def backwards(self, y):  # Add y parameter here
        learning_rate = 0.001
        momentum = 0.9
        # Store pre-activation values during forward pass
        self.z_values = [layer.z for layer in self.layers]
        
        # Calculate output layer gradients
        output_errors = (self.layers[-1].output - y) / len(y)
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
    data = np.array(data, dtype=np.float32)
    if data.size == 0:
        return np.zeros_like(data)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    return (data - data_min) / (data_max - data_min + 1e-8)


batch_length = int(input("batch length: "))
flattened_X = []
for j in range(batch_length):
    X = []
    for i in range(15):
        X.append(select("c, h, l, o, v", "AAPL", f"ID = {15 * j + i}"))
    flattened_X.append([value for sublist in X for inner_list in sublist for value in inner_list])#flattened, is one dimensional

y = []#expected outcomes
# Fix highNext15 calculation
highNext15 = []
for j in range(batch_length):
    current_high = -float('inf')
    for i in range(15):
        result = select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + 15 + i}")[0][0]
        current_high = max(current_high, result)
    highNext15.append(current_high)

# Fix high15 calculation
high15 = []
for j in range(batch_length):
    current_high = -float('inf')
    for i in range(15):
        result = select("o", "AAPL", f"ID = {int(flattened_X[j][0]) + i}")[0][0]
        current_high = max(current_high, result)
    high15.append(current_high)

# Fix target calculation
y = []
for j in range(batch_length):
    if highNext15[j] > high15[j]:
        y.append(1)
    else:
        y.append(0)

flattened_X = normalizeData(flattened_X[0])


n = Network(2, len(flattened_X), 2, flattened_X)

def testing():
    for i in range(1000):
        flattened_X = []
        for j in range(batch_length):
            X = []
            for k in range(15):
                X.append(select("*", "AAPL", f"ID = {15*(i+1) + k}"))
            #flattened_X.append([value for sublist in X for inner_list in sublist for value in inner_list])#flattened, is one dimensional
        #print(X)
        y = []#expected outcomes
        for j in X:
            #print(j)
            if select("o", "AAPL", f"ID = {int(j[0][0]) + 15}")[0][0] > float(j[0][4]) + (2 * abs((select("o", "AAPL", f"ID = {int(j[0][0])}")[0][0]) - (select("o", "AAPL", f"ID = {int(j[0][0]) + 1}")[0][0]))):
                val = 1
            elif select("o", "AAPL", f"ID = {int(j[0][0]) + 15}")[0][0] < float(j[0][4]) + (2 * abs((select("o", "AAPL", f"ID = {int(j[0][0])}")[0][0]) - (select("o", "AAPL", f"ID = {int(j[0][0]) + 1}")[0][0]))):
                val = 0
            else:
                try:
                    val = y[-1]
                except:
                    val = 0
            y.append(val)
        print(y)

        flattened_X.append([value for sublist in X for inner_list in sublist for value in inner_list])#flattened, is one dimensional
        flattened_X = normalizeData(flattened_X[0])
        n.newInput(flattened_X)

        cost = cost_calculation()
        print(cost.calculate(n.forward(), y))
        n.backwards()


testing()