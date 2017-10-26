import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import math


def linear_kernel(gamma, x, xi, offset, degree):
    """
    This function is responsible for calculating values for linear kernel method.
    :param gamma: the gamma value for rbf
    :param x: independent variable value
    :param xi: transpose of x
    :param offset: coef value
    :param degree: degree for polynomial
    :return: computed value for polynomial kernel
    """
    return np.dot(x,xi)


def polynomial_kernel(gamma, x, xi, offset, degree):
    """"
    This function is responsible for calculating values for polynomial kernel method.
    :param gamma: the gamma value for rbf
    :param x: independent variable value
    :param xi: transpose of x
    :param offset: coef value
    :param degree: degree for polynomial
    :return: computed value for linear kernel
    """
    return offset + ((np.dot(x, xi)) ** degree)


def gaussian_kernel(gamma, x, xi, offset, degree):
    """
    This function is responsible for calculating values for gaussian kernel method.
    :param gamma: the gamma value for rbf
    :param x: independent variable value
    :param xi: transpose of x
    :param offset: coef value
    :param degree: degree for polynomial
    :return: computed value for gaussian kernel
    """
    exponent = -np.sqrt(gamma * np.linalg.norm(x - xi) ** 2)
    return np.exp(exponent)


def get_kmatrix_values(kernel_function, xtrain, gamma, offset, degree):
    """
    This function is responsible for calculating the values for each row in matrix using kernel methods
    :param kernel_function: The kernel method (linear, polynomial, rbf)
    :param xtrain: Training dataset with independent values
    :param gamma: the gamma value from hyper parameters
    :param offset: the coefficient value from hyper parameters
    :param degree: the degree for polynomial
    :return: calculated k matrix
    """
    k_matrix = []
    for i in range(0,len(xtrain)):
        row_k_matrix=[]
        for j in range(0,len(xtrain)):
            row_k_matrix.append(kernel_function(gamma, xtrain[i], xtrain[j], offset, degree))
        k_matrix.append(row_k_matrix)

    k_matrix = np.array(k_matrix)
    return k_matrix


def plot_kernel_line(xtrain, ytrain, xtest, test_y, kr):
    """
    This function is responsible for plotting a graph
    """
    plt.scatter(xtrain[:, 0], ytrain[:, 0], c='r', label='data', zorder=1,edgecolors=(0, 0, 0))
    plt.plot(xtest[:, 0], test_y, c='g',label='KRR')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title(kr + ' Kernel Ridge Regression')
    plt.legend()
    plt.show()


def ridge_regression(kernel_method, xtrain, ytrain, xtest, ytest, gamma, offset, degree):
    """
    This function is responsible for:
    1. Calculating the values for K matrix using kernel methods
    2. Predicting the y values for testing data
    3. Call the 'performance_parameter' function the performance parameter values
    :param kernel_method: kernel method (linear, polynomial, rbf)
    :param xtrain: training dataset with independent parameters
    :param ytrain: training dataset with dependent parameter
    :param xtest: testing dataset with independent parameters
    :param ytest: testing dataset with dependent parameter
    :param gamma: the hyper parameter value gamma
    :param offset: coefficient value
    :param degree: degree (for polynomial)
    :return: the predicted values, accuracy
    """
    k_matrix = get_kmatrix_values(kernel_method, xtrain,gamma,offset,degree)
    gamma_matrix = np.dot(gamma, np.identity(len(k_matrix)))
    sec_entity = np.dot(inv(k_matrix + gamma_matrix), ytrain)
    y_pred = []

    for xi in xtest:
        k = np.array([kernel_method(gamma, x_i, xi,offset,degree) for x_i in xtrain])
        y = np.dot(k.T, sec_entity)
        y_pred.append(y)

    y_pred = np.array(y_pred)
    accuracy = performance_parameters(y_pred, ytest)
    return accuracy, y_pred


def get_folds(x_train, y_train):
    """
    This function is responsible for splitting the training data into 5 folds
    :param x_train: independent parameters' training data
    :param y_train: dependent parameter's training data
    :return:
    """
    kf = KFold(n_splits=5)
    folds=[]
    for k, (train, test) in enumerate(kf.split(x_train, y_train)):
        x_val = []
        y_val = []

        for index in train:
            x_val.append(x_train[index])
            y_val.append(y_train[index])
        train_data = [x_val, y_val]

        x_val = []
        y_val = []
        for index in test:
            x_val.append(x_train[index])
            y_val.append(y_train[index])
        test_data = [x_val, y_val]
        folds.append([train_data, test_data])
    return folds


def get_parameters_values(kernel_method, x_train, y_train):
    """
    This function is responsible for obtaining the hyper parameter values for each model
    :param kernel_method: The kernel method for which the hyper parameter required
    :param x_train: Training dataset
    :param y_train: Testing dataset
    :return: Values of coefficient and gamma
    """
    param_grid = {
        'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'coef0': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    kr = KernelRidge(kernel = kernel_method)
    grid = GridSearchCV(kr, param_grid)
    grid.fit(x_train, y_train)

    return (grid.best_estimator_.gamma, grid.best_estimator_.coef0)


def estimate(data):
    """
    This function is responsible for:
    1. Converting the data frame in two matrices of predictors and response.
    2. Split the data into training and testing data in 5 folds.
    3. For each fold, call the ridge regression function for computation.
    4. Get the hyper parameters values gamma, coef for each kernel method by calling 'get_parameter_values' function.
    5. Predict the values for testing data
    6. Plot the graph to show the ridge regression
    :param data: data set in frame
    :return: None
    """
    # Data frame of predictors
    df_x = (pd.DataFrame(data, columns=(['wheel_base','length','width','curb_weight','num_of_cylinders','engine_size','horsepower']))).as_matrix()
    # Data frame of target
    df_y = (pd.DataFrame(data, columns=(['price']))).as_matrix()

    # Split the data into training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=10)

    # Split the training data into 5 folds
    folds = get_folds(x_train,y_train)

    kernel=['linear','polynomial','rbf']

    # iterate through loop to run the implementation for various kernels.
    for kr in kernel:
        if kr == 'linear':
            kernel_method = linear_kernel
        elif kr == 'polynomial':
            kernel_method = polynomial_kernel
        else:
            kernel_method = gaussian_kernel
        (gamma, coef) = get_parameters_values(kr,x_train,y_train)
        for fold in folds:
           (accuracy, y_pred) = ridge_regression(kernel_method, fold[0][0], fold[0][1], fold[1][0], fold[1][1], gamma, coef, 1)

        # call the Ridge regression to predict values for test data
        (accuracy, y_pred) = ridge_regression(kernel_method, x_train, y_train, x_test, y_test, gamma, coef, 1)
        print('Variance score: %.2f' % accuracy)
        # Plot the regression graph
        # plot_kernel_line(x_train, y_train, x_test, y_pred,kr)


def performance_parameters(y_pred, y_test):
    """
    This function is used to calculate the measure of accuracy and prints the parameters of performance accuracy.
    :param y_pred: Predicted values of prices for testing data set
    :param y_test: Actual prices from testing data set.
    :return: variance score with respect to test and predicted values.
    """
    ss_res = np.sum(((y_test) - (y_pred)) ** 2)
    ss_tot = np.sum(((y_test) - (np.mean(y_test))) ** 2)
    r2 = 1-(ss_res / ss_tot)
    # print('Variance score: %.2f' % r2_score(y_test, y_pred))
    # print ("R^2: ", r2)
    return r2_score(y_test, y_pred)


def main():
    """
   1. Loads the data from final_auto which is preprocessed based on p and r values.
   2. Calls the estimate function which in turn implements K- fold and returns final results of model.
    """
    data = pd.read_csv('auto_final.csv')
    estimate(data)


if __name__ == '__main__':
   main()
