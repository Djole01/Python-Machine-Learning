import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
#print(data.head())

#select data attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())

#define what attribute we are trying to predict. This attribute is known as a label.
predict = "G3"

#array containing all features
x = np.array(data.drop([predict], 1))
#array containing labels
y = np.array(data[predict])

#We need to split our data into testing and training data.
#We will use 90% of our data to train and the other 10% to test.
#The reason we do this is so that we do not test our model on data that it has already seen.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

#training and scoring our model
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_) #values for each gradient
print('Intercept: \n', linear.intercept_) # This is the intercept

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    #model's prediction, the students data, and real grade he got
    print(predictions[x], x_test[x], y_test[x])




