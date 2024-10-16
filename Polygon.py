import polygon
from pprint import pp

client = polygon.RESTClient("DUEYmzwA2R9d8l5I18mNdycBZuHHYmXn")

aggs = []
for a in client.list_aggs(
    "AAPL",
    1,
    "minute",
    "2024-10-15",
    "2024-10-16",
    limit=5000,
):
    aggs.append(a)

pp(aggs)