import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller

class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None
        

    def ADF_Stationarity_Test(self, timeseries, printResults = True):
        #Dickey-Fuller test:  The Akaike Information Criterion (AIC) is used to determine the lag., BIC
        adfTest = adfuller(timeseries, regression='c', autolag='AIC')
        self.pValue = adfTest[1]
        if (self.pValue<self.SignificanceLevel):
            self.isStationary = True
        else:
            self.isStationary = False
        
        if printResults:
            dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
            #Add Critical Values
            for key,value in adfTest[4].items():
                dfResults['Critical Value (%s)'%key] = value

            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

series = pd.read_csv('daily-total-female-births.csv', header=0, index_col=0)
values = series.values
train = series_to_supervised(values, n_in=6)
trainX, trainy = train[:, :-1], train[:, -1]

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(trainX, trainy)

row = values[-6:].flatten()
yhat = model.predict(np.asarray([row]))
print('Input: %s, Predicted: %.3f' % (row, yhat[0]))

sTest = StationarityTests()
sTest.ADF_Stationarity_Test(series.values, printResults = True)
print("Is the time series stationary? {0}".format(sTest.isStationary))