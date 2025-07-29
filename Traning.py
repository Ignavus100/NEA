from TraningData import *
from NeuralNetwork import *
from DatabaseAccess import *
import pickle as pkl


def calcylateY(j, candles):
    future = select("c", "AAPL", "ID=" + str(20*j + 41))[0][0]
    present = select("c", "AAPL", "ID=" + str(20*j + 21))[0][0]
    percent = (future - present) / present
    if percent > 0.003:
        Y = [1, 0, 0] # buy
    elif percent < -0.003:
        Y = [0, 1, 0] # sell
    else:
        Y = [0, 0, 1] # neither
    return Y

def trainNN():
    with open("NN.pkl", "rb") as f:
        NN = pkl.load(f)
    epochs = int(input("Epochs: "))
    #NN = Network(150, 1, 2, 16)
    for j in range(epochs):
        for i in range(1505):
            for i in range(len(NN.layers)):
                NN.layers[i].cost = 0
            y = []
            y = np.array(y)
            data, candles = form_data(i)
            np.append(y, calcylateY(j, candles))
            NN.layers[0].activation = data
            print("data passed in")
            NN.layers[-1].activation = y
            print("Y values set")
            print("doing the forward pass")
            print(NN.forward())
            print("starting backpass")
            NN.backwards()
            print("backpass done")
            print("cost", NN.layers[0].cost)
            with open("NN.pkl", "wb") as f:
                pkl.dump(NN, f)

trainNN()