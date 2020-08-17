from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from math import sqrt
from matplotlib import pyplot
import numpy


def parse(x):
	return datetime.strptime(x, '%Y-%m-%d')


def forecastLstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)


def inverseDifference(history, yhat, interval=1):
	return yhat + history[-interval]
 
def fitLstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

def scale(train, test):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

def timeseriesToSupervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

 
sequence = read_csv('AMZN16-19.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parse)
 
rawData = sequence.values
differencedData = difference(rawData, 1)
 
slp = timeseriesToSupervised(differencedData, 1)
slpData = slp.values
 
trainingSet, testingSet = slpData[0:-250], slpData[-250:]
 
scaler, tsScaled, tScaled = scale(trainingSet, testingSet)
 
lstm_model = fitLstm(tsScaled, 1, 1500, 1)
train_reshaped = tsScaled[:, 0].reshape(len(tsScaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)
 
predictions = list()
for i in range(len(tScaled)):
	X, y = tScaled[i, 0:-1], tScaled[i, -1]
	yhat = forecastLstm(lstm_model, 1, X)
	yhat = invert_scale(scaler, X, yhat)
	yhat = inverseDifference(rawData, yhat, len(tScaled)+1-i)
	predictions.append(yhat)
	expected = rawData[len(trainingSet) + i + 1]
	print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
 
rmse = sqrt(mean_squared_error(rawData[-250:], predictions))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(rawData[-250:])
pyplot.plot(predictions)
pyplot.show()



 


