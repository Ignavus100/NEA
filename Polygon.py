import polygon
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import functools

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

class UpdateData():
    def __init__(self, currentBase, lowest, highest, bars):
        self.currentBase = currentBase
        self.lowest = lowest
        self.highest = highest
        self.bars = bars

    def NewBase(self, i):
        self.currentBase = i
    
    def NewLow(self, val):
        self.lowest = val

    def NewHigh(self, val):
        self.highest = val

    def NewBar(self, low, high):
        self.bars += 1
        self.lowest = low
        self.highest = high


def GraphData():

    #OC-Get user input
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

        #OC-checking the timeframe gaps
        if (aggs[data.currentBase].timestamp + 1000000) < currentPoint.timestamp:
            data.NewBase(i)
            data.NewBar(aggs[i].low, aggs[i].high)

        #OC-defining the upper and lower quartiles
        lq = min(aggs[data.currentBase].open, aggs[i].close)
        if lq == aggs[data.currentBase].open:
            bullish = False
        else:
            bullish = True
        uq = max(aggs[data.currentBase].open, aggs[i].close)

        #OC-making new bounds
        if data.lowest > aggs[i].low:
            data.NewLow(aggs[i].low)
        if data.highest < aggs[i].high:
            data.NewHigh(aggs[i].high)

        #OC-adding the bar to points in the position of how many bars there are
        try:
            points[data.bars] = [data.lowest, lq, aggs[data.currentBase].open, uq, data.highest, bullish]
        except:
            points.append([data.lowest, lq, aggs[data.currentBase].open, uq, data.highest, bullish])

        #OC-making the bars
        for j in range(len(points)):
            if points[j][5]:
                plt.boxplot(points[j][:5], positions=[j + 1], widths=0.8, patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor="red", color="red"),
                            whiskerprops=dict(color="red"),
                            capprops=dict(color="red"),
                            medianprops=dict(color="red"))

            else:
                plt.boxplot(points[j][:5], positions=[j + 1], widths=0.8, patch_artist=True, showfliers=False, 
                            boxprops=dict(facecolor="green", color="green"), 
                            whiskerprops=dict(color="green"), 
                            capprops=dict(color="green"), 
                            medianprops=dict(color="green"))

    #OC-label axis and graph
    plt.xlabel("Date and Time")
    plt.ylabel("Price")
    plt.title(f"{ticket} on {start}") 
    p=[]
    points = []
    data = UpdateData(0, aggs[0].low, aggs[0].high, 0)

    #OC-start the animation
    ani = FuncAnimation(plt.gcf(), animate, fargs=(data,), interval = 1000, frames = 1000, repeat = False)

    #OC-show the graph
    

    plt.show()

GraphData()