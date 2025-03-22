import numpy as np
import random
from DatabaseAccess import select

class Activation_ReLU:
    def forward(self, inputs):
        # Make sure inputs is a scalar or properly shaped array
        self.inputs = inputs
        self.output = max(0, inputs) if np.isscalar(inputs) else np.maximum(0, inputs)
        return self.output
        
    def backward(self, dvalues):
        # Handle scalar case
        if np.isscalar(self.inputs):
            self.dinputs = dvalues if self.inputs > 0 else 0
        else:
            self.dinputs = np.array(dvalues, copy=True)
            # Zero gradient where input values were negative
            self.dinputs[self.inputs <= 0] = 0
        return self.dinputs


class Activation_Softmax:
    def forward(self, inputs):
        # Ensure inputs is a numpy array
        inputs = np.array(inputs)
        self.inputs = inputs
        
        # Normalize inputs
        val = np.min(inputs)
        if val < 0:
            inputs = inputs - val
        
        # Calculate softmax properly
        exp_values = np.exp(inputs - np.max(inputs))
        probabilities = exp_values / np.sum(exp_values)
        
        self.output = probabilities
        return self.output
        
    def backward(self, dvalues):
        # Make sure dvalues is a numpy array
        dvalues = np.array(dvalues)
        
        # Create uninitialized array
        self.dinputs = np.zeros_like(dvalues)
        
        # Calculate gradient
        # For simplicity in this implementation, use a direct approach
        # This is a simplified gradient calculation for softmax
        # that works for classification tasks
        self.dinputs = dvalues
        
        return self.dinputs


class Node:
    def __init__(self, prevNeuronsAmount: int, val):
        self.val = val
        self.bias = 0.1 * random.randint(-50, 50)
        self.prevNeuronsAmount = prevNeuronsAmount
        self.weights = []
        if prevNeuronsAmount is not None:
            for i in range(prevNeuronsAmount):
                self.weights.append(0.1 * random.randint(-20, 20))
            self.weights = np.array(self.weights)
        else:
            self.weights = None
        self.inputs = None  # Store for backprop
        self.dweights = None  # Gradients for weights
        self.dbias = 0  # Gradient for bias
        self.activation = Activation_ReLU()
        
    def forward(self, prevLayer: list):
        # Store inputs for backprop
        self.inputs = [node.val for node in prevLayer]
        
        temp = 0
        for i in range(len(prevLayer)):
            if self.weights is not None:
                activation_output = self.activation.forward(self.bias + (self.weights[i] * float(prevLayer[i].val)))
            else:
                activation_output = 0
            temp += activation_output
        self.val = temp
        return self.val
        
    def backward(self, dvalue):
        # Initialize arrays if first backward pass
        if self.dweights is None and self.weights is not None:
            self.dweights = np.zeros_like(self.weights)
        
        # Gradient on weights
        if self.weights is not None:
            # Process gradient for each weight individually
            for i in range(len(self.weights)):
                # Gradient for this weight
                input_value = float(self.inputs[i])
                
                # Gradient through activation function (ReLU)
                dactivation = dvalue if self.activation.inputs > 0 else 0
                
                # Gradient on weight
                self.dweights[i] += dactivation * input_value
                
                # Gradient on bias
                self.dbias += dactivation
            
            # Gradient on values from previous layer
            self.dinputs = np.zeros(len(self.inputs))
            for i in range(len(self.weights)):
                # Gradient through activation function
                dactivation = dvalue if self.activation.inputs > 0 else 0
                
                # Gradient to the previous layer node
                self.dinputs[i] = dactivation * self.weights[i]
                
            return self.dinputs
        return None


class Network:
    def __init__(self, hidden: int, inp_size: int, out_size: int, values: list):
        self.hidden = hidden
        self.inp_size = inp_size
        self.out_size = out_size
        self.layers = []
        
        # Add learning rate parameter
        self.learning_rate = 0.01
        
        # Create Softmax activation for output layer
        self.activation_softmax = Activation_Softmax()

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
            
        # Store raw output for debugging
        self.raw_output = temp
        print("Raw output before softmax:", temp)
        
        output = self.activation_softmax.forward(temp)
        return output
        
    def backward(self, dvalues, y_true):
        # Backpropagate through the softmax activation
        d_output = self.activation_softmax.backward(dvalues)
        
        # Backpropagate through all layers starting from output layer
        dvalues_prev_layer = d_output
        
        # Iterate through layers backwards (except input layer)
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            next_dvalues = np.zeros(len(self.layers[i-1]))
            
            # Process each node in the layer
            for j, node in enumerate(layer):
                # Get gradient for this node
                if i == len(self.layers) - 1:
                    node_dvalue = dvalues_prev_layer[j]
                else:
                    node_dvalue = dvalues_prev_layer[j]
                
                # Backpropagate through the node
                dinputs = node.backward(node_dvalue)
                
                # Accumulate gradients for previous layer if we have them
                if dinputs is not None:
                    for k in range(len(dinputs)):
                        next_dvalues[k] += dinputs[k]
            
            # Set gradients for next iteration
            dvalues_prev_layer = next_dvalues
            
    def update_params(self):
        # Update weights and biases based on calculated gradients
        for i in range(1, len(self.layers)):
            for node in self.layers[i]:
                if node.weights is not None and node.dweights is not None:
                    # Update weights
                    for j in range(len(node.weights)):
                        node.weights[j] -= self.learning_rate * node.dweights[j]
                    
                    # Update bias
                    node.bias -= self.learning_rate * node.dbias
                    
                    # Reset gradients
                    node.dweights = np.zeros_like(node.weights)
                    node.dbias = 0


class cost:
    def calculate(self, output, y):
        cost = self.forward(output, y)
        return cost


class cost_calculation(cost):
    def forward(self, y_pred, y_true):
        # Convert inputs to numpy arrays for safer operations
        y_pred = np.array(y_pred)
        
        # Convert y_true to one-hot encoding
        one_hot = np.zeros(len(y_pred))
        one_hot[y_true[0]] = 1
        
        # Store for backprop
        self.y_pred = y_pred
        self.y_true = one_hot
        
        # Calculate MSE
        cost = np.mean(np.square(y_pred - one_hot))
        return cost
        
    def backward(self):
        # Gradient of MSE is 2 * (predicted - actual) / samples
        samples = len(self.y_pred)
        dvalues = 2 * (self.y_pred - self.y_true) / samples
        return dvalues


def normalizeData(data):
    normalized_data = []
    print("Data to normalize:", data)
    sigma = sum(data)
    for i in data:
        normalized_data.append(i/sigma if sigma != 0 else 0)
    return normalized_data


def train_network(network, X, y, epochs=1000):
    cost_function = cost_calculation()
    
    for epoch in range(epochs):
        # Forward pass
        predictions = network.forward()
        
        # Calculate loss
        loss = cost_function.calculate(predictions, y)
        
        # Backpropagation
        dvalues = cost_function.backward()
        network.backward(dvalues, y)
        
        # Update weights and biases
        network.update_params()
        
        # Print progress
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss}')
    
    return network


def main():
    # Get batch length
    batch_length = int(input("batch length: "))
    
    # Prepare data
    flattened_X = []
    for j in range(batch_length):
        X = []
        rnum = random.randint(1, 30125)
        for i in range(10):
            X.append(select("*", "AAPL", f"ID = {rnum + i}"))
        flattened_X.append([value for sublist in X for inner_list in sublist for value in inner_list])
    
    # Prepare labels
    y = []  # expected outcomes
    for i in flattened_X:
        if select("o", "AAPL", f"ID = {int(i[0]) + 15}")[0][0] > float(i[4]):
            val = 1
        elif select("o", "AAPL", f"ID = {int(i[0]) + 15}")[0][0] < float(i[4]):
            val = 0
        else:
            val = 1
        y.append(val)
    
    # Normalize data
    flattened_X_normalized = normalizeData(flattened_X[0])
    
    # Create network
    n = Network(2, len(flattened_X_normalized), 2, flattened_X_normalized)
    
    # Train network
    print("Starting training...")
    n = train_network(n, flattened_X_normalized, y, epochs=500)
    
    # Test network
    print("Final testing:")
    predictions = n.forward()
    print("Final predictions:", predictions)
    print("Expected:", y)


if __name__ == "__main__":
    main()