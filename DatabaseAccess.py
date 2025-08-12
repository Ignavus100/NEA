import polygon
import pyodbc as SQL
client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

def start():
    #initiate a cursor to execute SQL on the table
    SQL.lowercase = False
    #ensure that the db is connected as it is an accdb
    DB = SQL.connect(r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=E:\GitHub\NEA\Database\Main.accdb;")
    cur = DB.cursor()
    return cur, DB

def STOCK_insert(val, table):
    cur, DB = start()
    #create wanted values for the headings
    cur.execute(f"INSERT INTO {table}(c , h , l , o , t , v ) VALUES({val.close}, {val.high}, {val.low}, {val.open}, {val.timestamp}, {val.volume});")
    DB.commit()
    cur.close()
    DB.close()

def select(field, table, condition):
    cur, DB = start()
    #stores in variable as it has to be returned
    temp = cur.execute(f"SELECT {field} FROM {table} WHERE {condition};")
    temp = cur.fetchall()
    cur.close()
    DB.close()
    return temp

def delete(condition, table):
    cur, DB = start()
    cur.execute(f"DELETE FROM {table} WHERE {condition};")
    DB.commit()
    #clear #DELETED rows from the table to clean it up in access view
    query = f"DELETE FROM {table}"
    cur.execute(query)
    DB.commit()
    cur.close()
    DB.close()

def Create(name, values):
    cur, DB = start()
    #values are standard but there is a variable for it anyway
    cur.execute(f"CREATE TABLE {name}({values});")
    DB.commit()
    cur.close()
    DB.close()

def Custom(command, paramaters=None):
    cur, DB = start()
    #execute custom command
    if paramaters:
        cur.execute(command, paramaters)
    else:
        cur.execute(command)
    DB.commit()
    cur.close()
    DB.close()

# RUN ONLY ONECE PER STOCK TICKER!!!!!!!!!!!!!!!!!!!!!!!
start1 = "2024-01-01"
end = "2024-07-01"
#AAPL MSFT GOOGL AMZN FB TSLA NVDA INTC ADBE NFLX PYPL (tickers that i am using for the project)
ticket = "MSFT"
timeframe = "minute"

#OC-request from API
#OC-THE RUN ONCE CREATION OF THE AAPL DATABASE
""" for a in client.list_aggs(
    ticket,
    1,
    timeframe,
    start1,
    "1709151660000",
    limit=5000,
):
    #OC-insert the table values into the database or if needed create the database
    try:
        STOCK_insert(a, ticket)
    except:
        Create(ticket, "ID AUTOINCREMENT PRIMARY KEY, c DOUBLE, h DOUBLE, l DOUBLE, o DOUBLE, t DOUBLE, v DOUBLE")
        STOCK_insert(a, ticket)

 """
#CREATION OF THE USERS DATABASE RUN ONCE
#Create("UserData", "UserID AUTOINCRAMENT PRIMARY KEY, Forename VARCHAR, Surname VARCHAR, Username VARCHAR, HashedPassword INT, Balance INT")
#Create("UserTrades", "TradeID AUTOINCREMENT PRIMARY KEY, UserID INT, TradeOpen DOUBLE, TradeClose DOUBLE, AmountPlaced INT, ProfitMade INT")