from NeuralNetwork import *
from DatabaseAccess import *
import numpy as np

def Indicators(candles):
    output = []
    def SMA(candles):
        total = 0
        for i in range(len(candles)):
            total += candles[i][1]
        return (total/len(candles))
    
    def EMA(candles):
        closes = []
        span = 10
        multiplyer = 2/(span + 1)
        for i in range(len(candles)):
            closes.append(candles[i][1])
        ema = closes[0]
        for i in range(len(candles) - 1):
            ema.append((closes[i] - ema[i-1]) * multiplyer + ema[i-1])
        return ema[-1]
    
    def RSI(candles):
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
        HH = None
        LL = None
        for i in range(len(candles) - 1):
            if candles[i][1] > HH or HH == None:
                HH = candles[i][1]

            if candles[i][1] < LL or LL == None:
                LL = candles[i][1]
        SO = ((candles[-1][1] - LL) / (HH - LL)) * 100
        return SO
    
    def BB(candles):
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
        total = 0
        for i in range(len(candles) - 1):
            total = np.max((candles[i+1][2] - candles[i+1][3])  ,  (np.abs(candles[i+1][2] - candles[i][1]))  ,  (np.abs(candles[i+1][3] - candles[i][1]))) + total
        ATR = total / (len(candles) - 1)
        return ATR

    def ROC(candles):
        ROC = ((candles[-1][1] - candles[0][1]) / candles[0][1]) * 100
        return ROC
    
    def PC(candles):
        PC = (candles[-1][1] - candles[-2][1]) / candles[-2][1]
        return PC
    
    def LR(candles):
        LR = np.log10(candles[-1][1] / candles[-2][1])
        return LR
    
    output.append(SMA(candles))
    output.append(EMA(candles))
    output.append(RSI(candles))
    output.append(SO(candles))
    a, b = BB(candles)
    output.append(a)
    output.append(b)
    output.append(ATR(candles))
    output.append(ROC(candles))
    output.append(PC(candles))
    output.append(LR(candles))
    return output

def Normalise_data(data):
    highest_value = -9999999999999999999999999999999999999999999999999999999999
    for i in range(len(data)):
        if data[i] > highest_value:
            highest_value = data[i]
    for i in range(len(data)):
        data[i] = data[i] / highest_value

def form_data(iteration):
    final = []
    for i in range(20):
        temp1 = select("*", "AAPL", str("ID =" + 20 * iteration + i))
        temp2 = []
        for j in range():
            temp2.append(temp1[0][i])
        final.append(temp2)
    indicators = Indicators(final)
    for i in range(len(indicators)):
        final.append(indicators[i])
    final = Normalise_data(final)
    final = np.array(final)
    final = final.reshape(len(final), 1)

    return final