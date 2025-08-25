from NeuralNetwork import *
from DatabaseAccess import *
import numpy as np


def SMA(candles):
    #calculating the simple moving average
    total = 0
    for i in range(len(candles)):
        total += candles[i][1]
    return (total/len(candles))

def EMA(candles):
    #calculating the exponential moving average
    closes = []
    span = 10
    multiplyer = 2/(span + 1)
    for i in range(len(candles)):
        closes.append(candles[i][1])
    ema = [closes[0]]
    for i in range(len(candles) - 1):
        ema.append((closes[i] - ema[i-1]) * multiplyer + ema[i-1])
    return ema[-1]

def RSI(candles):
    #calculating the relative strength index
    print(candles)
    Gain = 0
    Loss = 0
    g = 0
    l = 0
    for i in range(len(candles) - 1):
        if candles[i+1][1] > candles[i][1]:
            Gain = Gain + candles[i+1][1] - candles[i][1]
            g += 1
        else:
            Loss = Loss + candles[i][1] - candles[i+1][1]
            l += 1
    avgGain = Gain/g
    avgLoss = Loss/l
    RS = avgGain/avgLoss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def SO(candles):
    #calculating the stochastic oscilator
    HH = None
    LL = None
    for i in range(len(candles) - 1):
        if HH == None or candles[i][1] > HH:
            HH = candles[i][1]

        if LL == None or candles[i][1] < LL:
            LL = candles[i][1]
    SO = ((candles[-1][1] - LL) / (HH - LL)) * 100
    return SO

def BB(candles, output):
    #calculating the bollinger bands
    total = 0
    for i in range(len(candles)):
        total += candles[i][1]
    mean = total/len(candles)
    deviation = []
    for i in range(len(candles)):
        deviation.append((candles[i][1] - mean)**2)
    total = 0
    for i in range(len(deviation)):
        total += deviation[i]
    variance = total / len(deviation)
    std_dev = variance ** (1/2)
    upper_band = output[0] + (2*std_dev)
    lower_band = output[0] - (2*std_dev)
    return upper_band, lower_band

def ATR(candles):
    #calculating the average true range
    total = 0
    for i in range(len(candles) - 1):
        total = np.max([(candles[i+1][2] - candles[i+1][3])  ,  (np.abs(candles[i+1][2] - candles[i][1]))  ,  (np.abs(candles[i+1][3] - candles[i][1]))]) + total
    ATR = total / (len(candles) - 1)
    return ATR

def ROC(candles):
    #calculating the rate of change
    ROC = ((candles[-1][1] - candles[0][1]) / candles[0][1]) * 100
    return ROC

def PC(candles):
    #calculating the price channel
    PC = (candles[-1][1] - candles[-2][1]) / candles[-2][1]
    return PC

def LR(candles):
    #calculating the linear regression
    LR = np.log10(candles[-1][1] / candles[-2][1])
    return LR


def Indicators(candles):
    #groupung all indicators to an array
    output = []
    output.append(SMA(candles))
    output.append(EMA(candles))
    output.append(RSI(candles))
    output.append(SO(candles))
    a, b = BB(candles, output)
    output.append(a)
    output.append(b)
    output.append(ATR(candles))
    output.append(ROC(candles))
    output.append(PC(candles))
    output.append(LR(candles))
    return output

def Normalise_data(data):
    #making all data between 0-1
    highest_value = -9999999999999999999999999999999999999999999999999999999999
    for i in range(len(data)):
        if data[i] > highest_value:
            highest_value = data[i]
    for i in range(len(data)):
        data[i] = data[i] / highest_value

    return data

def form_data(iteration):
    #creating a single array for every 20 candles which means that it can be passed into the NN
    final = []
    data = select("*", "AAPL", str("ID >=" + str(20 * iteration + 1) + "AND ID <=" + str(20 * iteration + 21)))
    for i in range(20):
        print(i)
        temp1 = data[i]
        temp2 = []
        for j in range(len(temp1)):
            temp2.append(temp1[j])
        final.append(temp2)
    candles = final
    indicators = []
    indicators = Indicators(final)
    final.append(indicators)
    for i in range(len(final)):
        final[i] = Normalise_data(final[i])
    temp = []
    for i in range(len(final)):
        for j in range(len(final[i])):
            temp.append(final[i][j])
    final = temp
    final = np.array(final)
    final = final.reshape(len(final), 1)

    return final, candles