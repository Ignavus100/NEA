import polygon
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use("dark_background")

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

def GraphData():

    # Fetch user input
    aggs = []
    start = "2024-10-20"
    end = "2024-10-21"
    ticker = input("Enter the ticker symbol you'd like to trade > ")
    timeframe = "hour"

    # Fetch data from API
    for a in client.list_aggs(
        ticker,
        1,
        timeframe,
        start,
        end,
        limit=5000,
    ):
        aggs.append(a)

    # Animation function to plot data at each frame
    def animate(i):
        if i >= len(aggs):
            return  # Stop when data is exhausted
        
        # Clear previous plot
        plt.cla()

        # Extract data for the current frame
        data = aggs[i]
        time = datetime.fromtimestamp(data.timestamp / 1000)  # Convert timestamp to datetime
        q1 = min(data.open, data.close)
        q3 = max(data.open, data.close)

        # Determine color and plot boxplot for current frame
        color = "green" if data.close > data.open else "red"
        plt.boxplot([data.low, q1, data.close, q3, data.high], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=color, color=color),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    medianprops=dict(color=color))

        # Set title and labels
        plt.title(f"{ticker} on {start} - {end}")
        plt.xlabel("Time")
        plt.ylabel("Price")

        # Annotate the time on the x-axis
        plt.xticks([1], [time.strftime('%Y-%m-%d %H:%M')])

    # Start animation
    ani = FuncAnimation(plt.gcf(), animate, interval=1000, frames=len(aggs), repeat=False)

    plt.tight_layout()
    plt.show()

GraphData()
