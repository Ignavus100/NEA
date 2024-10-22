import polygon
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use("dark_background")

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")
def GraphData():

    #OC-Get user input
    aggs = []
    start = "2024-10-20"
    end = "2024-10-21"
    ticket = input("the ticket you would like to trade > ")
    timeframe = "hour"

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
    def animate(i):
        plt.cla()
        data = aggs[i]
        p.append(data.timestamp)
        q1 = min(data.open, data.close)
        q3 = max(data.open, data.close)
        points.append([data.low, q1, data.close, q3, data.high])
        for i in range(len(points)):
            if aggs[i].open > aggs[i].close:
                plt.boxplot(points[i], positions=[p[i]], widths=3000000, patch_artist=True, showfliers=False, 
                            boxprops=dict(facecolor="red", color="red"), 
                            whiskerprops=dict(color="red"), 
                            capprops=dict(color="red"), 
                            medianprops=dict(color="red"))
            else:
                plt.boxplot(points[i], positions=[p[i]], widths=3000000, patch_artist=True, showfliers=False, 
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
    #OC-start the animation
    ani = FuncAnimation(plt.gcf(), animate, interval = 1000, frames = 500, repeat = False)

    #OC-show the graph
    plt.tight_layout()
    plt.show()

GraphData()