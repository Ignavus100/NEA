import sqlite3 as SQL

DB = SQL.connect("E:\\GitHub\\NEA\\Database\\Main.accdb")
cur = DB.cursor()
cur.execute("SELECT * From Table1")
rows = cur.fetchall()
DB.close()
for row in rows:
   print(row)

DB.close()