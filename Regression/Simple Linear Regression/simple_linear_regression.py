'''
__Author__="Gunasekar Jabbala"
__Email__="gunasekar.ai.dev@gmail.com"
'''

# Salary Prediction Model

# Import required libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

print(np.__version__)
print(mpl.__version__)
print(pd.__version__)

dataset = pd.read_csv('Salary_Data.csv')
print(dataset.describe())

print(dataset.info())
print(dataset.head(5))
print(dataset.tail(5))

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lr.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()