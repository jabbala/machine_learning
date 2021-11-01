'''
__Author__="Gunasekar Jabbala"
__Email__="gunasekar.ai.dev@gmail.com"
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
print(dataset.head())

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
lr_regressor = LinearRegression()
lr_regressor.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)
lr_regressor_2 = LinearRegression()
lr_regressor_2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lr_regressor.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lr_regressor_2.predict(poly_regressor.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lr_regressor_2.predict(poly_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lr_regressor.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lr_regressor_2.predict(poly_regressor.fit_transform([[6.5]]))