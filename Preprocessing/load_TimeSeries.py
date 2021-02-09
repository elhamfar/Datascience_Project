# load dataset using read_csv()
from pandas import read_csv
series = read_csv('TimeSeriesDataset.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
print(type(series))
print(series.head())