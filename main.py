import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&interval=1min&apikey=demo37QAPHW9NQALLU9I'
r = requests.get(url)
data = r.json()

print(data)