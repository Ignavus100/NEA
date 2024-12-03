#TODO sort the RSI graphing


import polygon
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from RSI import *

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

#OC-class to track data outside the animate function so it can be passed without complications
class UpdateData():
    def __init__(self, currentBase, lowest, highest, bars):
        self.currentBase = currentBase
        self.lowest = lowest
        self.highest = highest
        self.bars = bars
        self.__barsPos = []
        self.__Gain = []
        self.__Loss = []
        self.__FirstRun = True
        self.AvgGain = 0
        self.AvgLoss = 0
        self.xVal = []
        self.RSI = []

    def NewBase(self, i):
        self.currentBase = i
    
    def NewLow(self, val):
        self.lowest = val

    def NewHigh(self, val):
        self.highest = val

    def NewBar(self, low, high, time):
        self.__barsPos.append(time)
        self.bars += 1
        self.lowest = low
        self.highest = high

    def NewVal(self, low, high):
        self.__Loss.append(low)
        self.__Gain.append(high)

    def NewAvg(self):
        if self.__FirstRun:
            total = 0
            for i in self.__Gain:
                total += i
            self.AvgGain = total / 14
            self.__Gain = []

            total = 0
            for i in self.__Loss:
                total += i
            self.AvgLoss = total / 14
            self.__Loss = []

            self.__FirstRun = False
        
        else:
            self.AvgGain = (((self.AvgGain)*13) + self.__Gain[-1])/14
            self.AvgLoss = (((self.AvgLoss)*13) + self.__Loss[-1])/14

            self.__Loss = []
            self.__Gain = []


def GraphData():

    #OC-Get user input for bars
    aggs = []
    start = "2024-11-29"
    end = "2024-11-30"
    ticket = input("the ticket you would like to trade > ").upper()
    timeframe = "minute"

    #OC-request from API
    for a in client.list_aggs(
        ticket,
        1,
        timeframe,
        start,
        end,
        limit=5000,
    ):
        aggs.append(a)

    #OC-animation function that takes counter i and plots the data given from the API as a box plot
    def animate(i, data):
        #fig, (ax1, ax2) = plt.subplots(2)
        ax1.cla()
        ax2.cla()
  
        currentPoint = aggs[i]
        p.append(currentPoint.timestamp)

        #OC-making new average vals for the trendlines for iterations past the 14 threshold and graphing the RSI
        if i >= 14:
            RSI(data, i)
            ax2.plot(data.xVal, data.RSI)
            data.NewAvg()

        #OC-applying the new values for gain and loss for RSI
        if currentPoint.close > currentPoint.open:
            data.NewVal(0, (currentPoint.close - currentPoint.open))
        elif currentPoint.close < currentPoint.open:
            data.NewVal((currentPoint.open - currentPoint.close), 0)
        else:
            data.NewVal(0, 0)

        #OC-checking the timeframe gaps and creating a new bar if nececary
        if (aggs[data.currentBase].timestamp + 3600000) < currentPoint.timestamp:
            data.NewBase(i)
            data.NewBar(currentPoint.low, currentPoint.high, currentPoint.timestamp)

        #OC-defining the upper and lower quartiles
        lq = min(aggs[data.currentBase-1].close, currentPoint.close)
        if lq == aggs[data.currentBase-1].close:
            bullish = False
        else:
            bullish = True
        uq = max(aggs[data.currentBase -1 ].close, currentPoint.close)

        #OC-making new bounds
        if data.lowest > currentPoint.low:
            data.NewLow(currentPoint.low)
        if data.highest < currentPoint.high:
            data.NewHigh(currentPoint.high)

        #OC-adding the bar to points in the position of how many bars there are
        try:
            points[data.bars] = [data.lowest, lq, aggs[data.currentBase - 1].close, uq, data.highest, bullish]
        except:
            points.append([data.lowest, lq, aggs[data.currentBase - 1].close, uq, data.highest, bullish])

        #OC-making the bars the correct colour and size
        for j in range(len(points)):
            if points[j][5]:
                ax1.boxplot(points[j][:5], positions=[j + 1], widths=0.8, patch_artist=True, showfliers=True, showcaps=False, whis=(0, 100),
                            boxprops=dict(facecolor="#f23645", color="#f23645"),
                            whiskerprops=dict(color="#f23645"),
                            capprops=dict(color="#f23645"),
                            medianprops=dict(color="#f23645"))

            else:
                ax1.boxplot(points[j][:5], positions=[j + 1], widths=0.8, patch_artist=True, showfliers=True, showcaps=False, whis=(0, 100),
                            boxprops=dict(facecolor="#089981", color="#089981"), 
                            whiskerprops=dict(color="#089981"), 
                            capprops=dict(color="#089981"), 
                            medianprops=dict(color="#089981"))

    #OC-label axis and graph and adding colour
    fig, (ax1, ax2) = plt.subplots(2)
    #ax1._label("Date and Time")
    #ax1.ylabel("Price")
    #fig.title(f"{ticket} on {start}")
    #fig.gca().set_facecolor("#171b26")
    #fig.gcf().set_facecolor("#171b26")
    p=[]
    points = []
    data = UpdateData(0, aggs[0].low, aggs[0].high, 0)


    #OC-start the animation
    ani = FuncAnimation(fig, animate, fargs=(data,), interval = 1000, frames = 1000, repeat = False)

    #OC-show the graph
    plt.show()

#OC-run graphing function
GraphData()