import pandas as pd

data = pd.read_csv('file.txt', sep=" ", header=None)
data.columns = ["date", "entry", "count"]

print(data.groupby(['date']).agg('count')[['count']])