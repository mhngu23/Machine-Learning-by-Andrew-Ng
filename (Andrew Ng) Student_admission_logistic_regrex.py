'''The data used in this model is from Machine Learning course by Andrew Ng Practie Exercise 2: Logistic Regression
In this exercise, suppose you are the administration of a university department and you want to determine each applicants's chance
of admission based on their results on two exam
You have historical data from previous applicants that you can use as a training set for logistic regression. For each training
example, you have the applicant's scores on two exams and the admissions decision.'''
import pandas as pd
import numpy as np
#import plotting
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn import metrics
#Analysing and Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#reading the text file and save it as an csv file in the working directory
#text_file = pd.read_csv(r'C:\Users\Minh Hoang Nguyen\Desktop'
#                 r'\Machine Learning - Stanford\Coding Exercises'
#                 r'\Week 3 assignment logistic regression modelling\machine-learning-ex2\ex2\ex2data1.txt')
#text_file.to_csv(r'D:\PycharmProjects\Machine Learning\Logistic Regression Modeling\student-admission.csv',index=None)

#reading the data to work on
data = pd.read_csv(r'D:\PycharmProjects\Machine Learning\Logistic Regression Modeling\student-admission.csv',sep=',')

#print(data.head())

#Load data into python
data = data[['first_test','second_test','admission']] #this code is used to select multiple columns from the Dataframe
#you want to input a list of features (so in []) in  the [] for which representing that you want to take these columns out
predict = 'admission'
x = np.array(data.drop([predict],1))
y = np.array(data[predict])

#Part1: Plotting to visualize the data
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.
fig1 = plt.figure(1,figsize=(6,6))
ax1 = fig1.add_subplot(1,1,1)
colors = {1:'red', 0:'blue'}
style.use('default')
for i in range(len(y)):
    ax1.scatter(x[i, 0], x[i, 1], c=colors.get(y[i]))
ax1.set(aspect="equal", xlabel="Exam 1 Score", ylabel="Exam 2 Score")
plt.show()
input('Press enter to continue:')

#Part 2: Fitting a linear regression through our data set using
#this step is used to create a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#Part3: Make an instance of the model and fit our data in
logistic = LogisticRegression()
logistic.fit(x_train,y_train)

#Part4: Start predicting and output the accuracy of the model
predictions = logistic.predict(x_test)
#for i in range(len(x_test)):
#    print(x_test[i] , ':' , predictions[i])
score = logistic.score(x_test,y_test)
print('The accuracy of our model is', score)
input('Press enter to continue:')

#Part5: Testing the model on a student with test 1 score of 45 and test 2 score of 85
testing = np.array([45, 85]).reshape(1,-1)
testing_predictions_probability = logistic.predict_proba(testing)[:,1] #if dont have the [:,1] will return the probability of both classes
print('The admission probability for a student with test 1 score of 45 and test 2 score of 85',testing_predictions_probability)
input('Press enter to continue:')

#Part6: Plotting the decision boundary
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
h = 1  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #create a regtangular grid out of 2 given 1D array
Z = logistic.predict(np.c_[xx.ravel(), yy.ravel()]) #np.array return a flattened 1D array out of the 2 xx and yy grid
#then np.c_ will create a matrix out of those 2 1D array Ex: np.c_[(1,2,3),(4,5,6)] = ([1 4][2 5][3 6])
#then we use the logistic model to do the prediction on this which is the whole x value

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig2 = plt.figure(2,figsize=(6,6))
ax2 = fig2.add_subplot(1,1,1)
#f, ax = plt.subplots(figsize=(6, 6))
ax2.contour(xx, yy, Z, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
for i in range(len(y)):
    ax2.scatter(x[i, 0], x[i, 1], c=colors.get(y[i]))
ax2.set(aspect="equal", xlabel="Exam 1 Score", ylabel="Exam 2 Score")
plt.show()
input('Press enter to continue:')

#Part7: Create a confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
print('This is the confusion matrix using metrics function from sklearn')
print(cm)
input('Press enter to continue:')

fig3 = plt.figure(3,figsize=(6,6))
ax3 = fig3.add_subplot(1,1,1)
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
ax3.set(aspect="equal", xlabel="Actual label", ylabel="Predicted label")
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
print('This is the confusion matrix using seaborn')
plt.show()

