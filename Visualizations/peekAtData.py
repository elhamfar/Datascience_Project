from pandas import Series
series = Series.from_csv('TimeSeriesDataset.csv', header=0)
print(series.head(10))