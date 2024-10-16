import requests
from pprint import pp

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=CFG&interval=30min&apikey=demo37QAPHW9NQALLU9I&outputsize=full'
r = requests.get(url)
data = r.json()

pp(data)