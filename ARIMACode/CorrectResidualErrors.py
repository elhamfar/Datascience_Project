# summarize residual errors for an ARIMA model
from pandas import Series
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true)*100)
# load data
series = Series.from_csv('BiWeeklyData.csv',header=0)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.86)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # predict
 model = ARIMA(history, order=(0,2,1))
 model_fit = model.fit(disp=0)
 yhat = model_fit.forecast()[0]
 predictions.append(yhat)
 # observation
 obs = test[i]
 history.append(obs)
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()


# summarize residual errors from bias corrected forecasts
from pandas import Series
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# load data
series = Series.from_csv('mdata2.csv')
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.84)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
bias = -10.526363
for i in range(len(test)):
 # predict
 model = ARIMA(history, order=(3,2,0))
 model_fit = model.fit( disp=0)
 yhat = bias + float(model_fit.forecast()[0])
 predictions.append(yhat)
 # observation
 obs = test[i]
 history.append(obs)
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
mape = mean_absolute_percentage_error(test, predictions)
print('RMSE: %.3f' % rmse)
print('MAPE: %.3f' % mape)
# summarize residual errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot residual errors
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(211)
#residuals.plot(kind='kde', ax=pyplot.gca())
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()