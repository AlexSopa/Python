#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
#Loading the data set
dataset = loadtxt('exams.csv', delimiter=',')
#Split into input(x) and output(y) variables

X = dataset[:,0:8]
Y = dataset[:,8]
#define thee keras model
model = Sequential()
#https://www.tensorflow.org/api_docs/python/tf/keras/Model
#Under this, the first hidden layer has 12 nodes and uses the relu activation function
model.add(Dense(12, input_dim=8,activation='relu')) #input_dim=8 means that the model expects rows of data with 8 variables
#The second layer underneath this has 8 nodes and uses the same function
#Relu gives better performance.
model.add(Dense(8, activation='relu'))
#The output layer has 1 node & uses the sigmoid activation function
#Sigmoid is the output layer because we need to ensure the network output is between 0 and 1
#Sigmoid makes our ouput easier to map
#'Sigmoid Smushification Function' takes the who plot of data and squishes it down to values between 0-1
model.add(Dense(1, activation='sigmoid'))
#Compile the Keras model
#We use 'adam' because it is a popular version of gradient descent because 
#it automatically tunes itself & gives good results in a wide range of problems
#To learn more about the Adam version of the stochastic gradient descent see 
#'https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/'
#Define Stochastic : 
#randomly determined; having a random probability distribution or pattern that may be 
#analyzed statistically but may not be predicted precisely.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Epoch: One pass through of all rows in the training data set
#Batch : One or more samples considered by the model within
#an epoch before weights are updated
#Adam is the quickest way to get to the local minima (Least amunt of computing power but benifits from having more training & predicting)
#Adam makes guesses and takes larger leaps.

#fitting the model to the data set
model.fit(X,Y, epochs=100, batch_size=10) #150 pass thru, 150 samples from the file
#Evaluate function will return a list with two values, the first will be the loss of the model
#on the dataset and the second will be the accuracy of the model.
#Will only be reporting acuracy
_, accuracy = model.evaluate(X,Y)
print('Accuracy: %.2f' % (accuracy*100))
#Tie it all together
#If you get an error while running, change following lines
#model.fit(X,Y, epochs=150, batch_size=10, verbose=0)
#_, accuracy = model.evaluate(X,Y, verbose=0)


#MAKING PREDICTIONS
#Make probability predictions with model
predictions = model.predict_classes(X)
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
