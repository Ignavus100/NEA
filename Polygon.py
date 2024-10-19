import polygon
from pprint import pp
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
from datetime import datetime

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

aggs = []
for a in client.list_aggs(
    "AAPL",
    1,
    "minute",
    "1729172815270",
    "1729173013946",
    limit=5000,
):
    aggs.append(a)

def animate(i):
    #plt.cla()

    data = aggs[i]
    x = data.timestamp
    y = data.open

    plt.plot(x, y, marker = ".")
    y = data.close
    plt.plot(x, y, marker = ".")

ani = FuncAnimation(plt.gcf(), animate, interval = 1000, frames = 500, repeat = False)
#fig, ax = plt.subplots()
#for i in range(len(aggs)):
 #   ax.plot(aggs[i].timestamp, aggs[i].close, markersize=100)
plt.tight_layout()
plt.show()
pp(aggs[1].open)