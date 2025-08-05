import matplotlib.pyplot as plt

def RSI_1(data, xVal):
    try:
        data.RSI.append(100 - (100/(1+(data.AvgGain / data.AvgLoss))))
    except:
        data.RSI.append(100)
    data.xVal.append(xVal)