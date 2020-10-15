'''The data used in this model is from Machine Learning course by Andrew Ng Practie Exercise 1: Linear Regression
In this exercise, suppose you are the CEO of a restaurant and are considering different cities
You want to use this data to select where to expand next
The first column from our data set is the population of the city and the second column is the profit of a food truck in that city'''
#importing packages for linear regression
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

#importing packages for plotting
import matplotlib.pyplot as pyplot
from matplotlib import style

#importing packages to save the model
import pickle


#reading the text data file and save it as an csv file in the working directory
#text_file = pd.read_csv(r'C:\Users\Minh Hoang Nguyen\Desktop'
#                 r'\Machine Learning - Stanford\Coding Exercises\Week 2 assignment linear regression modelling'
#                 r'\machine-learning-ex1\ex1\ex1data1.txt')
#text_file.to_csv(r'D:\PycharmProjects\Machine Learning\Linear Regression Modelling\food-truck.csv',index=None)
data = pd.read_csv(r'D:\PycharmProjects\Machine Learning\Linear Regression Modelling\food-truck.csv', sep=',')
#Load data into python
data = data[['population','profit']]
x = np.array(data['population']).reshape(-1,1)
y = np.array(data['profit'])

#Part 1: Plotting the data
"""Before getting start on any task you would want to visualizing the data"""
style.use('default')
pyplot.scatter(x,y)
pyplot.xlabel('population of City in $10000')
pyplot.ylabel('profit in $10000')
pyplot.show()
input('Press enter to continue:')

#Part 2: Fitting a linear regression through our data set using
#this step is used to create a training set and a test set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
accuracy = linear.score(x_test,y_test)
with open('foodtruckmodel.pickle','wb') as f:
    pickle.dump(linear,f)
pickle_in = open('foodtruckmodel.pickle','rb')
linear = pickle.load(pickle_in)
print('The accuracy of our model is',accuracy)
print("The coefficient is: \n", linear.coef_) #print out the calculated coefficient for each featuers
print("The intercept is: \n", linear.intercept_) #print out the y intercept of the model





