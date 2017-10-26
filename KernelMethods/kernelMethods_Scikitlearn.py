#Import libraries

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


#read data into dataframe
data = pd.read_csv("C://Users//VISHAL//PycharmProjects//MLAss1/auto.csv")

# Selecting specific columns to implement
df_x = (pd.DataFrame(data, columns=(['wheel_base', 'length', 'width', 'num_of_cylinders', 'curb_weight', 'engine_size',
                                   'horsepower']))).as_matrix()
df_y = (pd.DataFrame(data, columns=(['price']))).as_matrix()

#Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=10)

#Sci-kit learn KernelRidge Regression model with linear kernel method
kr = GridSearchCV(KernelRidge(kernel='linear', gamma=0.1), cv=5,
                  param_grid={"gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                              "coef0": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

print(kr)
kr.fit(x_train,y_train)

y_kr = kr.predict(x_test)

print('Coef:', kr.best_estimator_.coef0)
print('Gamma:', kr.best_estimator_.gamma)
print("Score: " + str(kr.score(x_test, y_test)))

#Plot the graph for linear kernel
plt.scatter(x_train[:, 0], y_train[:, 0], c='r', label='data', zorder=1,
                edgecolors=(0, 0, 0))

plt.plot(x_test[:, 0], y_kr, c='g',
             label='KRR')
plt.xlabel('data')
plt.ylabel('target')
plt.title( 'Linear Kernel Ridge Regression')
plt.legend()
plt.show()

#Sci-kit learn KernelRidge Regression model with Gaussian kernel method
kr = GridSearchCV(KernelRidge(kernel='polynomial', alpha=1.0, gamma=0.1, degree=1), cv=5,
                 param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                             "gamma": np.logspace(-2, 2, 5),
                             "coef0": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1,2,3,4,5,6,7,8,9,10]})

print(kr)
kr.fit(x_train,y_train)

y_kr = kr.predict(x_test)
print('Coef:', kr.best_estimator_.coef0)
print('Gamma:', kr.best_estimator_.gamma)
print("Score: " + str(kr.score(x_test, y_test)))

#Plot the graph for Polynomial kernel
plt.scatter(x_train[:, 0], y_train[:, 0], c='r', label='data', zorder=1,
                edgecolors=(0, 0, 0))

plt.plot(x_test[:, 0], y_kr, c='g',
             label='KRR')
plt.xlabel('data')
plt.ylabel('target')
plt.title( 'Polynomial Kernel Ridge Regression')
plt.legend()
plt.show()

#Sci-kit learn KernelRidge Regression model with Gaussian kernel method
kr =GridSearchCV( KernelRidge (kernel='rbf',gamma=0.1), cv=5,
                  param_grid = {"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                "gamma":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                "coef0": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
print(kr)
kr.fit(x_train,y_train)

y_kr = kr.predict(x_test)
print('Coef:', kr.best_estimator_.coef0)
print('Gamma:', kr.best_estimator_.gamma)
print("Score: " + str(kr.score(x_test, y_test)))


#Plot the graph for Gaussian kernel
plt.scatter(x_train[:, 0], y_train[:, 0], c='r', label='data', zorder=1,
                edgecolors=(0, 0, 0))

plt.plot(x_test[:, 0], y_kr, c='g',
             label='KRR')
plt.xlabel('data')
plt.ylabel('target')
plt.title( 'Gaussian Kernel Ridge Regression')
plt.legend()
plt.show()