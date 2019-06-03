from keras.models import Sequential
from keras.layers import Dense
import numpy
import random

# split into input (X) and output (Y) variables
X1 = numpy.arange(0, 10, 0.01) # 0 to 10 with 0.01 interval 
Y1 = numpy.sin(X1)
D = []
for i in range(0,len(X1)):
    D.append([X1[i],Y1[i]]) # [value, sin(value)] matrix
    
random.shuffle(D)
X=[]
Y=[]
for i in range(0, len(D)):
    X.append(D[i][0])
    Y.append(D[i][1])
    
# create model
model = Sequential()

# Dense implements the operation: output = activation(dot(input, kernel) + bias) 
# where activation is the element-wise activation function passed as the activation argument, 
# kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer 
# (only applicable if use_bias is True).

model.add(Dense(200, input_dim=1, activation='relu')) 
# input_dim = 1: 1 input parameter
# 200: 200 neurons in the first hidden layer
# relu: rectified linear activation unit (if input > 0 return input; else return 0)

model.add(Dense(16, activation='sigmoid'))
# 16: 16 neurons in the second hidden layer
# sigmoid: S-shape function that exists between 0 and 1. Used for models where we need to predict
# the probability of an output (probabilities exist between 0 and 1)
# f(x) =    _____L_______  L = curve's maximum value     x0 = value of sigmoid's midpoint
#          1+e^-(k(x-x0))  k = steepness of the curve
# a standard logistic function is called sigmoid function (k=1, x0=0, L=1)
# S(x) = __1___        
#        1+e^-x     (S- shaped curve)
# This curve has a finite limit of:
# ‘0’ as x approaches −∞
# ‘1’ as x approaches +∞
# The output of sigmoid function when x=0 is 0.5
# If output is more than 0.5 , we can classify the outcome as 1 (or YES) and if it is less than 0.5 , we can classify it as 0(or NO) .

model.add(Dense(1, activation='linear'))
#output will not be confined between any range.

# Compile model
# SGD = Stochastic gradient descent optimizer
# Stochastic = uses randomly selected samples to evaluate the gradients
# Gradient descent = 1st order iterative optimization algorithm for finding the minimum of a function.
# It tries to find a local minimum taking steps proportional to the negative of the gradient of the function at the current point.
# Gradient = rate of inclination or declination of a slope (descending in this case)
# So SGD approximates the gradient descent in a stochastic way.
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mean_squared_error'])
# mean squared error: average of the squared differences between the predicted and actual values

import matplotlib.pyplot as plt
#Xt = numpy.arange(0.0,12.0,0.4)
Xt = X1

# Fit the model and shows the result
for i in range(1,100):

    # example has 1000 samples (iterations*batch_size=1000)
    # the smaller the batch_size, the more iterations will have to be done in each epoch
    model.fit(X, Y, epochs=10, batch_size=50, verbose=0)
    # epochs=10: the entire dataset passes the neural network forward and backward 10 times.
    # if there are few epochs, every iteration will yield very little training and the function won't have 
    # an acceptable approximation in time. We would have to increase the number of iterations.
    # batch_size=20: 20 samples will be passed through the network at once

    Yt = numpy.sin(Xt)
    predictions = model.predict(Xt) 
    sai = []
    for pred in predictions:
        sai.append(pred[0])
    plt.clf()
    plt.plot(Xt, Yt, 'b', Xt, sai, 'r')
    plt.ylabel('Y / Predicted Value '+str(i))
    plt.xlabel('X Value')
    plt.draw()
    plt.pause(0.001)