import polygon
from DatabaseAccess import select
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from RSI import *
import pickle as pkl
import numpy as np
from TraningData import *
from NeuralNetwork import *
from DatabaseAccess import *

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")
'''
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

'''
#def GraphData(canvas, Fig, ax1, ax2, i, data, points, frame_width):
'''
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
'''
'''    if i > frame_width:
        aggs = select("*", "AAPL", f"ID <= {i} AND ID > {i-frame_width}")
    else:
        aggs = select("*", "AAPL", f"ID <= {i+1}")
    #OC-animation function that takes counter i and plots the data given from the API as a box plot
    def animate(i, data):
        #fig, (ax1, ax2) = plt.subplots(2)
        #OC - creating the buy sell or remain signals
        signal = "n"
        if i % 20 == 0 and i != 0:
            with open("NN_protect.pkl", "rb") as f:
                NN = pkl.load(f)
            NN_data, candles = form_data(i)
            NN.layers[0].activation = NN_data
            output, cost = NN.forward()
            print(output)
            if output.activation[0] == np.max(output):
                signal = "b"
            elif output.activation[1] == np.max(output):
                signal = "s"

        currentPoint = aggs[i % frame_width]
        p.append(currentPoint[5])

        #OC-making new average vals for the trendlines for iterations past the 14 threshold and graphing the RSI
        if i >= 14:
            RSI_1(data, i)
            ax2.plot(data.xVal, data.RSI, color="#089981")
            data.NewAvg()

        #OC-applying the new values for gain and loss for RSI
        if currentPoint[1] > currentPoint[4]:
            data.NewVal(0, (currentPoint[1] - currentPoint[4]))
        elif currentPoint[1] < currentPoint[4]:
            data.NewVal((currentPoint[4] - currentPoint[1]), 0)
        else:
            data.NewVal(0, 0)

        #OC-making a new bar
        data.NewBase(i)
        data.NewBar(currentPoint[3], currentPoint[2], currentPoint[5])
        #if (aggs[data.currentBase][5] + 36000) < currentPoint[5]:

        #OC-defining the upper and lower quartiles
        lq = min(aggs[data.currentBase-1][1], currentPoint[1])
        if lq == aggs[data.currentBase-1][1]:
            bullish = False
        else:
            bullish = True
        uq = max(aggs[data.currentBase -1 ][1], currentPoint[1])

        #OC-making new bounds
        if data.lowest > currentPoint[3]:
            data.NewLow(currentPoint[3])
        if data.highest < currentPoint[2]:
            data.NewHigh(currentPoint[2])

        #OC-adding the bar to points in the position of how many bars there are
        try:
            points[data.bars] = [data.lowest, lq, aggs[data.currentBase - 1][1], uq, data.highest, bullish]
        except:
            points.append([data.lowest, lq, aggs[data.currentBase - 1][1], uq, data.highest, bullish])

        #OC-making the bars the correct colour and size
        print(points)
        ax1.boxplot(points[j][:5], positions=[1])'''
'''for j in range(len(points)):
            if signal == "n":
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
            else:
                if signal == "s":
                    ax1.boxplot(points[j][:5], positions=[j + 1], widths=0.8, patch_artist=True, showfliers=True, showcaps=False, whis=(0, 100),
                                boxprops=dict(facecolor="#ffffff", color="#ffffff"),
                                whiskerprops=dict(color="#ffffff"),
                                capprops=dict(color="#ffffff"),
                                medianprops=dict(color="#ffffff"))

                else:
                    ax1.boxplot(points[j][:5], positions=[j + 1], widths=0.8, patch_artist=True, showfliers=True, showcaps=False, whis=(0, 100),
                                boxprops=dict(facecolor="#000000", color="#000000"), 
                                whiskerprops=dict(color="#000000"), 
                                capprops=dict(color="#000000"), 
                                medianprops=dict(color="#000000"))
                    '''
'''return data, points

    #OC-label axis and graph and adding colour
    #ax1._label("Date and Time")
    #ax1.ylabel("Price")
    #fig.title(f"{ticket} on {start}")
    #fig.gca().set_facecolor("#171b26")
    #fig.gcf().set_facecolor("#171b26")
    p=[]
    
    if i == 0:
        data = UpdateData(0, aggs[0][3], aggs[0][2], 0)
        points = []


    #OC-start the animation
    #ani = FuncAnimation(Fig, animate, fargs=(data,), interval = 1000, frames = 1000, repeat = False)
    data, points = animate(i, data)

    #OC-show the graph
    canvas.draw()
    return data, points
'''
#OC-run graphing function
#GraphData()

def graph(frame_width, end_of_frame, canvas, main_axis):
    main_axis.clear()
    main_axis.set_xlim(0, frame_width + 0.5)
    candle_points = select("l, c, o, h", "AAPL", f"ID <= {end_of_frame} AND ID > {end_of_frame - frame_width}")

    def create_candle(position_in_frame, absoloute_position, candle_points, Max):
        candle_points = candle_points[Max - position_in_frame - 1]
        plotting_points = []
        plotting_points.append(candle_points[0])
        plotting_points.append(min(candle_points[1], candle_points[2]))
        plotting_points.append(candle_points[1])
        plotting_points.append(max(candle_points[1], candle_points[2]))
        plotting_points.append(candle_points[3])

        signal = "n"
        if absoloute_position % 20 == 0 and absoloute_position != 0:
            with open("NN_protect.pkl", "rb") as f:
                NN = pkl.load(f)
            NN_data, candles = form_data(absoloute_position)
            NN.layers[0].activation = NN_data
            output, cost = NN.forward()
            print(output)
            if output.activation[0] == np.max(output):
                signal = "b"
            elif output.activation[1] == np.max(output):
                signal = "s"

        if candle_points[1] > candle_points[2]:
            bullish = True
        else:
            bullish = False

        return signal, plotting_points, bullish
    
    for i in range(min(frame_width, end_of_frame)):
        signal, plotting_points, bullish = create_candle(i, end_of_frame - i, candle_points, min(frame_width, end_of_frame))
        #if signal == "n":
        if bullish:
            main_axis.boxplot(plotting_points, positions=[min(frame_width, end_of_frame)-i], widths=0.8, patch_artist=True, showfliers=True, showcaps=False, whis=(0, 100),
                                boxprops=dict(facecolor="#f23645", color="#f23645"),
                                whiskerprops=dict(color="#f23645"),
                                capprops=dict(color="#f23645"),
                                medianprops=dict(color="#f23645"))
        else:
            main_axis.boxplot(plotting_points, positions=[min(frame_width, end_of_frame)-i], widths=0.8, patch_artist=True, showfliers=True, showcaps=False, whis=(0, 100),
                                boxprops=dict(facecolor="#089981", color="#089981"), 
                                whiskerprops=dict(color="#089981"), 
                                capprops=dict(color="#089981"), 
                                medianprops=dict(color="#089981"))


    canvas.draw()