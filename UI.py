import customtkinter as tk
from CTkMessagebox import CTkMessagebox
from Graphing import *
from DatabaseAccess import *
from Login import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from time import sleep


#DATA STORE_____________________START
class DataStore:
    def __init__(self):
        self.i = 0
        self.frame_width = 15
        self.paused = True
        self.buying = False
        self.selling = False
        self.start_price = 0
        self.amount = 0
        self.UserID = 0
        self.temp_end = 0
        self.balance = 10000000
        self.started = False

I = DataStore()
#DATA STORE_____________________END




#FRAMES_________________________________________________________________________START
class Main(tk.CTk):
    def __init__(self, **kwargs):
        tk.CTk.__init__(self, **kwargs)

        width= self.winfo_screenwidth() 
        height= self.winfo_screenheight()
        self.geometry(f"{width}x{height}")
        self.title("Trading Simulator")

        self.frames = {}
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        for frame in (login, sign_up, TradingPannel):#all neccecary frames
            f = frame(self)
            f.grid(row=0, column=0, sticky="nsew")
            self.frames[frame] = f
        self.switch(login)#set starting frame for program

    def switch(self, frame):
        self.frames[frame].lift()

    def callback(self):
        if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
            self.destroy()




class TradingPannel(tk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        tk.CTkFrame.__init__(self, master, **kwargs)
    
        #_______________TOOLBAR_______________START
        toolbar = tk.CTkFrame(self)

        button1 = tk.CTkButton(toolbar, text="Next candle", command=lambda: plot(canvas1, fig1, ax1, I.frame_width, True))
        button1.pack(pady=12, padx=10, side="right")

        button2 = tk.CTkButton(toolbar, text="Play", command=lambda: toggle_play_pause(self, button2, canvas1, fig1, ax1, I.frame_width, True))
        button2.pack(pady=12, padx=10, side="right")
        
        slider = tk.CTkSlider(toolbar, from_=2, to=100, command=lambda value: change_frame_width(value, canvas1, fig1, ax1))
        slider.pack(pady=12, padx=10, side="right")

        sliderLBL = tk.CTkLabel(toolbar,  text="Frame Width")
        sliderLBL.pack(pady=12, padx=4, side="right")

        button3 = tk.CTkButton(toolbar, 24, text="->", command=lambda: move_left(canvas1, fig1, ax1))
        button3.pack(pady=12, padx=5, side="right")

        button4 = tk.CTkButton(toolbar, 24, text="<-", command=lambda: move_right(canvas1, fig1, ax1))
        button4.pack(pady=12, padx=5, side="right")

        button5 = tk.CTkButton(toolbar, text="Buy", command=lambda: toggle_buy_stop(button5, entry1.get(), amountLBL))
        button5.pack(pady=12, padx=10, side="left")

        button6 = tk.CTkButton(toolbar, text="Sell", command=lambda: toggle_sell_stop(button6, entry1.get(), amountLBL))
        button6.pack(pady=12, padx=10, side="left")

        label1 = tk.CTkLabel(toolbar, text="£")
        label1.pack(pady=12, padx=0, side="left")

        entry1 = tk.CTkEntry(toolbar, placeholder_text="Amount Placed")
        entry1.pack(padx=10, pady=12, side="left")

        amountLBL = tk.CTkLabel(toolbar, text=f"Balance: £{I.balance / 100}")
        amountLBL.pack(pady=12, padx=4, side="left")

        indicators = ["1", "2", "3", "4"]
        dropdown = tk.CTkComboBox(toolbar, values=indicators, width=200)
        dropdown.set("select an indicator")
        dropdown.pack(pady=3, padx=15)

        button7 = tk.CTkButton(toolbar, text="Load Indicator", command=lambda: load_indicator(dropdown.get(), canvas2, ax2, self))
        button7.pack(pady=3, padx=10)

        toolbar.pack(side="top", fill="x")
        #_______________TOOLBAR_______________END



     



        #______________________GRAPHING THE CANDLES____________________ START
        fig1, (ax1) = plt.subplots(1)
        canvas1 = FigureCanvasTkAgg(fig1, self)
        plot(canvas1, fig1, ax1, I.frame_width, True)


        #______________________UPDATE______________________ START
        reload_trading(self, canvas1, fig1, ax1, amountLBL)
        #______________________UPDATE______________________ END


        canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #______________________GRAPHING THE CANDLES____________________ END




        #______________________GRAPHING THE INDICATORS____________________ START
        fig2, (ax2) = plt.subplots(1)
        canvas2 = FigureCanvasTkAgg(fig2, self)


        #______________________UPDATE______________________ START
        load_indicator(self, canvas2, fig2, ax2, amountLBL)
        #______________________UPDATE______________________ END


        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #______________________GRAPHING THE INDICATORS____________________ END
        



class login(tk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        tk.CTkFrame.__init__(self, master, **kwargs)

        label = tk.CTkLabel(self, text="Login")
        label.pack(pady=12, padx=10)

        entry1 = tk.CTkEntry(self, placeholder_text="Username")
        entry1.pack(pady=12, padx=10)

        entry2 = tk.CTkEntry(self, placeholder_text="Password", show="*")
        entry2.pack(pady=12, padx=10)

        button1 = tk.CTkButton(self, text="Login", command=lambda: login_logic(entry1.get().upper(), entry2.get(), master))
        button1.pack(pady=12, padx=10)

        button2 = tk.CTkButton(self, text="Sign up", command=lambda: master.switch(sign_up), fg_color="grey17", hover_color="grey20")
        button2.pack(pady=12, padx=10)

        button3 = tk.CTkButton(self, text="Cheat", command=lambda: master.switch(TradingPannel))
        button3.pack(pady=12, padx=10)




class sign_up(tk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        tk.CTkFrame.__init__(self, master, **kwargs)

        label = tk.CTkLabel(self, text="Sign Up")
        label.pack(pady=12, padx=10)

        entry1 = tk.CTkEntry(self, placeholder_text="First Name")
        entry1.pack(pady=12, padx=10)

        entry2 = tk.CTkEntry(self, placeholder_text="Surname")
        entry2.pack(pady=12, padx=10)

        entry3 = tk.CTkEntry(self, placeholder_text="Password (8-12 chr)", show="*")
        entry3.pack(pady=12, padx=10)

        entry4 = tk.CTkEntry(self, placeholder_text="Password (check)", show="*")
        entry4.pack(pady=12, padx=10)

        button1 = tk.CTkButton(self, text="Sign up", command=lambda: SignUp_logic(entry1.get(), entry2.get(), entry3.get(), entry4.get()))
        button1.pack(pady=12, padx=10)

        button2 = tk.CTkButton(self, text="Login", command=lambda: master.switch(login), fg_color="grey17", hover_color="grey20")
        button2.pack(pady=12, padx=10)


#FRAMES_________________________________________________________________________END













#ALL THE LOGIC TO THE BUTTONS ETC.____________________________________________________________________________________START


def login_logic(username, password, master):
    Hpassword = hash(password)
    if len(select("username", "UserData", f"username = '{str(username)}' AND HashedPassword = {Hpassword}")) == 1:
        master.switch(TradingPannel)
        data = select("UserID, Balance, i", "UserData", f"username = '{str(username)}' AND HashedPassword = {Hpassword}")
        I.UserID = data[0][0]
        I.balance = data[0][1]
        I.i = data[0][2]
        I.temp_end = data[0][2]
        I.started = True
    else:
        CTkMessagebox(title="Error", message="The username or password is incorrect")



def SignUp_logic(Forename, Surname, password1, password2):
    if password1 != password2:
        CTkMessagebox(title="Error", message="The passwords do not match")
    elif len(password1) < 8 or len(password1) > 12:
        CTkMessagebox(title="Error", message="Passwords must be between 8 and 12 characters long")
    elif Forename == "" or Surname == "":
        CTkMessagebox(title="Error", message="All fields must be filled")
    else:
        number = str(len(select("Forename", "UserData", f"Forename LIKE '{Forename[0]}{Surname[0]}%'")) + 1)
        username = f"{Forename[0].upper()}{Surname[0].upper()}{number}"
        Hpassword = hash(password1)
        command = "INSERT INTO UserData (Forename, Surname, Username, HashedPassword) VALUES(?, ?, ?, ?, ?);"
        peramaters = (f"{Forename}", f"{Surname}", f"{username}", f"{Hpassword}", 100000)
        Custom(command, peramaters)
        CTkMessagebox(title="Message", message=f"Make note of your username {username}, you will need this to login.")



def stop(fig):
    Custom(f"UPDATE UserData SET i = {I.i} WHERE UserID = {I.UserID}")
    plt.close(fig)


def plot(canvas, fig, ax1, frame_width, incrament):
    if incrament:
        I.i += 1
        I.temp_end += 1
    graph(frame_width, I.i, canvas, ax1)
    stop(fig)



def play(root, canvas, fig, ax1, frame_width, incrament):
    if not I.paused:
        plot(canvas, fig, ax1, I.frame_width, incrament)
        root.after(5000, lambda: play(root, canvas, fig, ax1, frame_width, incrament))



def change_frame_width(value, canvas, fig, ax1):
    I.frame_width = int(value)
    plot(canvas, fig, ax1, I.frame_width, False)



def move_left(canvas, fig, ax1):
    I.temp_end += 1
    if I.temp_end <= I.i and I.temp_end >= I.frame_width:
        graph(I.frame_width, I.temp_end, canvas, ax1)
    stop(fig)



def move_right(canvas, fig, ax1):
    I.temp_end -= 1
    if I.temp_end <= I.i and I.temp_end >= I.frame_width:
        graph(I.frame_width, I.temp_end, canvas, ax1)
    stop(fig)



def toggle_play_pause(root, button2, canvas, fig, ax1, frame_width, incrament):
    if I.paused:
        I.paused = False
        button2.configure(text="Pause")
        return play(root, canvas, fig, ax1, frame_width, incrament)
    else:
        I.paused = True
        button2.configure(text="Play")



def toggle_buy_stop(button, amount, amountLBL):
    try:
        if float(amount) % 1 == 0:
            amount = float(amount) * 100
        elif (float(amount) * 10) % 1 == 0:
            amount = float(amount) * 10
        elif (float(amount) * 100) % 1 == 0:
            amount = float(amount) * 100
        else:
            amount = int(amount)* 100
    except:
        amount = int(amount)
    if amount == "" or float(amount) % 1 != 0:
        CTkMessagebox(title="Error", message="amount must be an integer")
    elif amount > I.balance:
        CTkMessagebox(title="Error", message="Insufficient funds")
    else:
        if not I.buying and not I.selling:
            try:
                I.buying = True
                I.start_price = select("c", "AAPL", f"ID = {I.i + 1}")[0][0]
                I.amount = int(amount)
                button.configure(text="Stop")
            except:
                pass
        elif I.buying ^ I.selling:
            try:
                end_price = select("c", "AAPL", f"ID = {I.i + 1}")[0][0]
                profit = int(I.amount + (I.amount * ((end_price - I.start_price) / I.start_price)))
                Custom("INSERT INTO UserTrades (UserID, TradeOpen, TradeClose, AmountPlaced, ProfitMade) VALUES (?, ?, ?, ?, ?);", (I.UserID, I.start_price, end_price, I.amount, profit))
                I.buying = False
                button.configure(text="Buy")
                amount_penny = I.balance
                amount_penny += profit
                I.balance = amount_penny
                amountPound = amount_penny / 100
                Custom(f"UPDATE UserData SET Balance = {I.balance} WHERE UserID = {I.UserID}")
                amountLBL.configure(text=f"Balance: £{str(amountPound)}")
            except:
                pass



def toggle_sell_stop(button, amount, amountLBL):
    try:
        if float(amount) % 1 == 0:
            amount = float(amount) * 100
        elif (float(amount) * 10) % 1 == 0:
            amount = float(amount) * 10
        elif (float(amount) * 100) % 1 == 0:
            amount = float(amount) * 100
        else:
            amount = int(amount)* 100
    except:
        amount = int(amount)
    if amount == "" or float(amount) % 1 != 0:
        CTkMessagebox(title="Error", message="amount must be an integer")
    elif amount > I.balance:
        CTkMessagebox(title="Error", message="Insufficient funds")
    else:
        try:
            if not I.selling and not I.buying:
                I.selling = True
                I.start_price = select("c", "AAPL", f"ID = {I.i}")[0][0]
                I.amount = amount
                button.configure(text="Stop")
            else:
                end_price = select("c", "AAPL", f"ID = {I.i}")[0][0]
                profit = int((I.amount * ((I.start_price - end_price) / I.start_price)))
                Custom("INSERT INTO UserTrades (UserID, TradeOpen, TradeClose, AmountPlaced, ProfitMade) VALUES (?, ?, ?, ?, ?);", (I.UserID, I.start_price, end_price, I.amount, profit))
                I.selling = False
                button.configure(text="Sell")
                amount_penny = I.balance
                amount_penny += profit
                I.balance = amount_penny
                amountPound = amount_penny / 100
                Custom(f"UPDATE UserData SET Balance = {I.balance} WHERE UserID = {I.UserID}")
                amountLBL.configure(text=f"Balance: £{str(amountPound)}")
        except:
            pass



def reload_trading(root1, canvas, fig, ax1, amountLBL):
    if I.started:
        plot(canvas, fig, ax1, I.frame_width, False)
        amountLBL.configure(text=f"Balance: £{I.balance / 100}")
        I.started = False
    root1.after(100, lambda: reload_trading(root1, canvas, fig, ax1, amountLBL))



def load_indicator(indicator, canvas, ax1, self, ax2=None):
    if indicator == "select an indicator":
        CTkMessagebox(title="Error", message="You must select an option.")
    else:
        pass
    #TODO: logic for the graphing of the indicators in the Graphing.py file




#ALL THE LOGIC TO THE BUTTONS ETC.____________________________________________________________________________________END












#START THE PROGRAM
root = Main()
root.mainloop()