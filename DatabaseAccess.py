import polygon
import pyodbc as SQL
db = "https://ignavus100.github.io/NEA_API/Main.accdb"
client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

def start():
    SQL.lowercase = False
    DB = SQL.connect(r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=E:\GitHub\NEA\Database\Main.accdb;")
    cur = DB.cursor()
    return cur, DB

def insert(val, table):
    cur, DB = start()
    cur.execute(f"INSERT {val.close}, {val.high}, {val.low}, {val.open}, {val.timestamp}, {val.volume} INTO {table};")
    DB.commit()
    DB.close()

def select(field, table):
    cur, DB = start()
    temp = cur.execute(f"SELECT {field} FROM {table};")
    temp = cur.fetchall()
    DB.close()
    return temp

def delete(condition, table):
    cur, DB = start()
    cur.execute(f"DELETE FROM {table} WHERE {condition};")
    DB.commit()
    query = f"DELETE FROM {table}"
    cur.execute(query)
    DB.commit()
    DB.close()

def Create(name, values):
    cur, DB = start()
    cur.execute(f"CREATE TABLE {name}({values});")
    DB.commit()
    DB.close()


start1 = "2024-11-01"
end = "2024-11-02"
ticket = input("aapl").upper()
timeframe = "minute"

#OC-request from API
for a in client.list_aggs(
    ticket,
    1,
    timeframe,
    start1,
    end,
    limit=5000,
):
    try:
        insert(a, ticket)
    except:
        Create(ticket, "ID AutoNumber, c int, h int, l int, o int, t int, v int")
        insert(a, ticket)