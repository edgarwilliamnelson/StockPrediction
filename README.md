Project to gain experience with Deep Learning. 

The dataset is Amazon’s stock price history from 2016-2019. The network is trained on the first two years and then tested on the third. 

The data is first prepared for consumption by transforming the time series into a supervised learning problem, manipulated to be stationary, then scaled to fit with the networks activation function. 

The LSTM(Long-Short-Term-Memory) network used is set to use a single batch with 1500 epochs and a single neuron. 

The Performance over the testing portion of data is shown below, with the predicted values being shown in yellow and expected in blue. 

![Alt text](Performance.png?raw=true "Optional Title")


Dependencies:  
Python 2 or 3   
Keras with Tensorflow backend   
Pandas  
NumPy  
Matplotlib  
scikit-learn.  





