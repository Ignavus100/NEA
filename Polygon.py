import polygon
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import functools

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

class UpdateData():
    def __init__(self, currentBase, lowest, highest):
        self.currentBase = currentBase
        self.lowest = lowest
        self.highest = highest

    def NewBase(self, i):
        self.currentBase = i
    
    def NewLow(self, val):
        self.lowest = val

    def NewHigh(self, val):
        self.highest = val


def GraphData():

    #OC-Get user input
    aggs = []
    start = "1726095600000"
    end = "1726105600000"
    ticket = input("the ticket you would like to trade > ")
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
        print(data.currentBase)
        print(data.lowest)
        print(data.highest)
        plt.cla()
        currentPoint = aggs[i]
        p.append(currentPoint.timestamp)

        #OC-checking the timeframe gaps
        try:
            if (aggs[data.currentBase].timestamp + 3600000) < currentPoint.timestamp:
                data.NewBase(i)
        except:
            pass

        #OC-defining the upper and lower quartiles
        uq = min(aggs[data.currentBase].open, currentPoint.close)
        lq = max(aggs[data.currentBase].open, currentPoint.close)

        #OC-making new bounds
        if data.lowest > aggs[i].low:
            data.NewLow(aggs[i].low)
        if data.highest < aggs[i].high:
            data.NewHigh(aggs[i].high)

        #OC-adding the bar to points in the position of how many bars there are
        try:
            points[data.currentBase] = [data.lowest, uq, aggs[data.currentBase].open, lq, data.highest]
        except:
            points.append([data.lowest, uq, aggs[data.currentBase].open, lq, data.highest])

        #OC-making the bars
        for j in range(data.currentBase + 1):
            if aggs[data.currentBase].open > currentPoint.close:
                plt.boxplot(points[data.currentBase], positions=[aggs[data.currentBase].timestamp], widths=3000000, patch_artist=True, showfliers=False, 
                            boxprops=dict(facecolor="red", color="red"), 
                            whiskerprops=dict(color="red"), 
                            capprops=dict(color="red"), 
                            medianprops=dict(color="red"))
            else:
                plt.boxplot(points[data.currentBase], positions=[aggs[data.currentBase].timestamp], widths=3000000, patch_artist=True, showfliers=False, 
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
    data = UpdateData(0, aggs[0].low, aggs[0].high)

    #OC-start the animation
    ani = FuncAnimation(plt.gcf(), animate, fargs=(data,), interval = 1000, frames = 500, repeat = False)

    #OC-show the graph
    plt.show()

GraphData()