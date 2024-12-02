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
        self.__High = []
        self.__Low = []
        self.__AvgCount = 0
        self.__AvgContents = []

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
        #TODO make it so that the highs and lows go into arrays
        self.__AvgHigh = ((self.__AvgHigh * self.__AvgCount) + high)/(self.__AvgCount+1)
        self.__AvgLow = ((self.__AvgLow * self.__AvgCount) + low)/(self.__AvgCount + 1)
        self.__AvgCount += 1

    def NewAvg(self):
        #TODO calculate the average (called every 14 updates) and update the RSI based on the formula
        self.__AvgHigh.sort()
        high = self.__AvgHigh[-1]


def GraphData():

    #OC-Get user input for bars
    aggs = []
    start = "2024-10-20"
    end = "2024-10-21"
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
        plt.cla()
        currentPoint = aggs[i]
        p.append(currentPoint.timestamp)

        #OC-making new average vals for the trendlines
        if i % 14 == 0:
            data.NewVal(aggs[i].low, aggs[i].high)
        else:
            data.NewAvg()

        #OC-checking the timeframe gaps and creating a new bar if nececary
        if (aggs[data.currentBase].timestamp + 3600000) < currentPoint.timestamp:
            data.NewBase(i)
            data.NewBar(aggs[i].low, aggs[i].high, aggs[i].timestamp)

        #OC-defining the upper and lower quartiles
        lq = min(aggs[data.currentBase-1].close, aggs[i].close)
        if lq == aggs[data.currentBase-1].close:
            bullish = False
        else:
            bullish = True
        uq = max(aggs[data.currentBase -1 ].close, aggs[i].close)

        #OC-making new bounds
        if data.lowest > aggs[i].low:
            data.NewLow(aggs[i].low)
        if data.highest < aggs[i].high:
            data.NewHigh(aggs[i].high)

        #OC-adding the bar to points in the position of how many bars there are
        try:
            points[data.bars] = [data.lowest, lq, aggs[data.currentBase - 1].close, uq, data.highest, bullish]
        except:
            points.append([data.lowest, lq, aggs[data.currentBase - 1].close, uq, data.highest, bullish])

        #OC-making the bars the correct colour and size
        for j in range(len(points)):
            if points[j][5]:
                plt.boxplot(points[j][:5], positions=[j + 1], widths=0.8, patch_artist=True, showfliers=True, showcaps=False, whis=(0, 100),
                            boxprops=dict(facecolor="#f23645", color="#f23645"),
                            whiskerprops=dict(color="#f23645"),
                            capprops=dict(color="#f23645"),
                            medianprops=dict(color="#f23645"))

            else:
                plt.boxplot(points[j][:5], positions=[j + 1], widths=0.8, patch_artist=True, showfliers=True, showcaps=False, whis=(0, 100),
                            boxprops=dict(facecolor="#089981", color="#089981"), 
                            whiskerprops=dict(color="#089981"), 
                            capprops=dict(color="#089981"), 
                            medianprops=dict(color="#089981"))

    #OC-label axis and graph and adding colour
    plt.xlabel("Date and Time")
    plt.ylabel("Price")
    plt.title(f"{ticket} on {start}")
    plt.gca().set_facecolor("#171b26")
    plt.gcf().set_facecolor("#171b26")
    p=[]
    points = []
    data = UpdateData(0, aggs[0].low, aggs[0].high, 0)

    #OC-start the animation
    ani = FuncAnimation(plt.gcf(), animate, fargs=(data,), interval = 1000, frames = 1000, repeat = False)

    #OC-show the graph
    plt.show()

#OC-run graphing function
GraphData()