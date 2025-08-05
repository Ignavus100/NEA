import customtkinter as tk
from CTkMessagebox import CTkMessagebox
from Graphing import *
from DatabaseAccess import *
from Login import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataStore:
    def __init__(self):
        self.i = 1
I = DataStore()
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
        frame_width = 15

        #the code for graphing in tkinter --------------- START
        fig, (ax1) = plt.subplots(1)
        canvas = FigureCanvasTkAgg(fig, self)
        GDData, GDPoints = plot(canvas, fig, ax1, None, [], [], frame_width)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #the code for graphing in tkinter --------------- END

        button1 = tk.CTkButton(self, text="Next candle", command=lambda: plot(canvas, fig, ax1, None, GDData, GDPoints, frame_width))
        button1.pack(pady=12, padx=10)


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


def login_logic(username, password, master):
    Hpassword = hash(password)
    if len(select("username", "UserData", f"username = '{str(username)}' AND HashedPassword = {Hpassword}")) == 1:
        master.switch(TradingPannel)
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
        command = "INSERT INTO UserData (Forename, Surname, Username, HashedPassword) VALUES(?, ?, ?, ?);"
        peramaters = (f"{Forename}", f"{Surname}", f"{username}", f"{Hpassword}")
        Custom(command, peramaters)
        CTkMessagebox(title="Message", message=f"Make note of your username {username}, you will need this to login.")

def stop(fig):
    plt.close(fig)


def plot(canvas, fig, ax1, ax2, GDData, GDPoints, frame_width):
    Temp_points = []
    
    if len(GDPoints) > frame_width:
        for i in range(frame_width):
            Temp_points.append(None)
        for i in range(frame_width):
            Temp_points[frame_width - i - 1] = GDPoints[-i]
    else:
        Temp_points = GDPoints

    graph(frame_width, I.i, canvas, ax1)
    I.i += 1
    stop(fig)
    return GDData, GDPoints


#tk.set_appearance_mode("system")
#tk.set_default_color_theme("blue")

#root = tk.CTk()
#width= root.winfo_screenwidth() 
#height= root.winfo_screenheight()
#root.geometry("500x500")

#login()
root = Main()
root.mainloop()