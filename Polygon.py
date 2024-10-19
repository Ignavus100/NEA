import polygon
from pprint import pp
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import numpy as np
from datetime import datetime

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

aggs = []
start = "2024-10-17"
end = "2024-10-18"
for a in client.list_aggs(
    "AAPL",
    1,
    "hour",
    start,
    end,
    limit=5000,
):
    aggs.append(a)

#plt.xlim(0, 10)
p = []
def animate(i):
    #plt.cla())
    data = aggs[i]
    p.append(data.timestamp)
    q1 = min(data.open, data.close)
    q3 = max(data.open, data.close)
    x = datetime.fromtimestamp(int(data.timestamp)/1000)
    plt.boxplot([data.low, q1, data.close, q3, data.high], positions=[data.timestamp], widths=3600000)
    
#    x = datetime.fromtimestamp(int(data.timestamp)/1000)
#    y = data.open

#    plt.plot(x, y, marker = ".")
#    y = data.close
#    plt.plot(x, y, marker = ".")
#    plt.fill_between(x, data.open, data.close)

plt.xlabel("Date and Time")
plt.ylabel("Price")
plt.title(f"APPL on {start}")
ani = FuncAnimation(plt.gcf(), animate, interval = 1000, frames = 500, repeat = False)

#for i in range(len(aggs)):
 #   ax.plot(aggs[i].timestamp, aggs[i].close, markersize=100)
plt.tight_layout()
plt.show()