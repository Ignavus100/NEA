from TraningData import *
from NeuralNetwork import *
from DatabaseAccess import *
import pickle as pkl

def calcylateY(data, j, candles):
    h = select("c", "AAPL", "ID=" + str(20*j + 20))[0][0]
    upper, lower = BB(candles, [SMA(candles)])
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
    epochs = int(input("Epochs: "))
    for j in range(epochs):
        for i in range(1505):
            y = []
            y = np.array(y)
            data, candles = form_data(i)
            np.append(y, calcylateY(data, j, candles))
            NN.layers[0].activation = data
            NN.y = y
            NN.backwards()
            print(NN.layers[-1].cost)
    with open("NN.pkl", "wb") as f:
        pkl.dump(NN, f)

runNN()