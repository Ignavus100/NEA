import pyodbc as SQL

def start():
    SQL.lowercase = False
    DB = SQL.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=E:\GitHub\NEA\Database\Main.accdb;')
    cur = DB.cursor()
    return cur, DB

def insert(val, table):
    cur, DB = start()
    cur.execute(f"INSERT {val} INTO {table}")
    DB.close()

def select(field, table):
    cur, DB = start()
    return cur.execute(f"SELECT {field} FROM {table}")
    DB.close()

def delete(condition, table):
    cur, DB = start()
    cur.execute(f"DELETE FROM {table} WHERE {condition}")
    DB.close()