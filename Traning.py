from TraningData import *
from NeuralNetwork import *
from DatabaseAccess import *
import pickle as pkl

def calcylateY(data):
    h = select("c", "AAPL", "ID=" + str(data[20][0] + 20))[0]
    upper, lower = BB(data, SMA(data))
    if h > upper:
        Y = 1
    elif h < lower:
        Y = 0
    else:
        Y = 0.5
    return Y

def runNN():
    with open("NN.pkl", "rb") as f:
        NN = pkl.load(f)
    epochs = input("Epochs: ")
    for j in range(epochs):
        for i in range(1505):
            y = []
            y = np.array(y)
            data = form_data(i)
            y.append(calcylateY(data))
            NN.layers[0] = data
            NN.y = y
            NN.backwards()
    with open("NN.pkl", "wb") as f:
        pkl.dump(NN, f)