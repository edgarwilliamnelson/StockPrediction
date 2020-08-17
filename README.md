Project to gain experience with Deep Learning. Explored procedures of data transformation to be readied for network consumption and development of a rudimentary LSTM (Long-Short-Term-Memory) model.  

The dataset is Amazon’s stock price history from 2016-2019, a stock’s price history is essentially a pre formatted time-series dataset that will hopefully contain some easily recognizable trends  

The network is trained on the first two years of data and then tested on the third.  
The data is first prepared for consumption by transforming the time series into a supervised learning problem, manipulated to be stationary, then scaled to fit with the networks activation function.  

The LSTM network used is set to use a single batch with 1500 epochs and a single neuron. 
The performance over the testing portion of data is shown below, with the predicted values being shown in yellow and expected in blue. 


![Alt text](Performance.png?raw=true "Optional Title")


Dependencies:  
Python 2 or 3   
Keras with Tensorflow backend   
Pandas  
NumPy  
Matplotlib  
scikit-learn.  







