from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_csv(filename):
    """
        This function is responsible for loading the data in a list.
        :param source: Source data set
        :return: Loaded data set in a list
        """
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def convert_to_float(dataset, column):
    """
       This function is responsible for converting string columns to float.
       """
    for row in dataset:
        row[column] = float(row[column].strip())


def convert_to_int(dataset, column):
    """
       This function is responsible for converting string columns to integer.
       """
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def dataset_find_minmax(dataset):
    """
    This function is responsible to find the min and max values for each column in the dataset
    :param dataset: source dataset
    :return:
    """
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


def normalize_scale(dataset, minmax):
    """
    This function is responsible for normalizing/ scaling the dataset to the range of 0-1
    :param dataset: source dataset
    :param minmax: minimun maximum value for each coulmn in the dataset
    :return:
    """
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def cross_validation_split(dataset, number_of_folds):
    """
    This function is resplosible for splitting the dataset into train and test data for each fold
    :param dataset: source dataset
    :param n_folds: number of folds
    :return: splitted data into folds
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / number_of_folds)
    for i in range(number_of_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric_evaluation(actual, predicted):
    """
       This function is responsible for computing the accuracy of predicted values against actual values.
       :param actual_values: Actual Y values
       :param predicted_values: Predicted Y values
       :return: Accuracy percentage for a given fold.
       """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def training_algorithm(dataset, number_of_folds, learning_rate, epoch, no_of_hidden):
    """
       This function is responsible for:
          1. Creating 5 folds in which row values are added based on an index of data set which is generated randomly.
          2. For splitting the data into two sets training data set and testing data set.
          3. Calling the function 'back_propagation_algo' for training the algorithm.
          4. Calling the accuracy function for figuring the actual accuracy results.
       """
    folds = cross_validation_split(dataset, number_of_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = back_propagation_algo(train_set, test_set, learning_rate, epoch, no_of_hidden)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric_evaluation(actual, predicted)
        scores.append(accuracy)
    return scores


def neuron_activate(weights, inputs):
    """
    This function is responsible for calculating the weighted sum of inputs.
    :param weights: weights
    :param inputs: inputs
    :return: calculated value for each input
    """
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def neuron_transfer(activation):
    """
    This function is responsible for calculating the sigmoid of activation
    :param activation: activation
    :return: sigmoid value
    """
    return 1.0 / (1.0 + exp(-activation))


def forward_propagation(network, row):
    """
    This function is responsible for forward propagation for a row of data from our dataset with our neural network.
    """
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = neuron_activate(neuron['weights'], inputs)
            neuron['output'] = neuron_transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def calculate_derivative(output):
    """
    This function is responsible for calculating the derivative of the neuron output
    """
    return output * (1.0 - output)


def calculate_backward_propagation_error(network, expected):
    """
    This function is responsible for calculating the back propagation error and store it in neurons i.e
    error is calculated between expected outputs and outputs forward propagated from the network
    """
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * calculate_derivative(neuron['output'])


def update_weights(network, row, learning_rate):
    """
    This function is responsible for updating the weights after calculating the errors for
    neuron in the network via back propagation method.
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']


def train_network(network, train, learning_rate, epoch, no_of_outputs):
    """
    This function is responsible for training network using stochastic gradient descent.
    This involves multiple iterations of exposing a training dataset to the network and
    for each row of data forward propagating the inputs, backpropagating the error and
    updating the network weights.
    """
    for epoch in range(epoch):
        for row in train:
            outputs = forward_propagation(network, row)
            expected = [0 for i in range(no_of_outputs)]
            expected[row[-1]] = 1
            calculate_backward_propagation_error(network, expected)
            update_weights(network, row, learning_rate)


def initialize_network(no_of_inputs, no_of_hidden, no_of_outputs):
    """
    This function is responsible for creating a new neural network which is ready for training
    :param no_of_inputs: number of inputs
    :param no_of_hidden: number of neurons to have in the hidden layer
    :param no_of_outputs: number of outputs
    :return: list of network
    """
    network = list()
    hidden_layer = [{'weights': [random() for i in range(no_of_inputs + 1)]} for i in range(no_of_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(no_of_hidden + 1)]} for i in range(no_of_outputs)]
    network.append(output_layer)
    return network


def predict_values(network, row):
    """
    This function is responsible for predicting the values for validation data as well as test data
    """
    outputs = forward_propagation(network, row)
    return outputs.index(max(outputs))


def back_propagation_algo(train, test, learning_rate, epoch, no_of_hidden):
    """
    This function is responsible for back propagation with Stochastic Gradient Descent.
    Calls the initialize network for creating a new neural network.
    Calls train network for training the network with forward propagation, backpropagating with errors
    and updating network weights.
    :param train: the training data
    :param test: the testing data
    :param learning_rate: Limit the amount each coefficient is corrected each time it is updated
    :param epoch:  number of times to run through training data to update weights.
    :param no_of_hidden: number of hidden neurons
    :return:
    """
    no_of_inputs = len(train[0]) - 1
    no_of_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(no_of_inputs, no_of_hidden, no_of_outputs)
    train_network(network, train, learning_rate, epoch, no_of_outputs)
    predictions = list()
    for row in test:
        prediction = predict_values(network, row)
        predictions.append(prediction)
    return (predictions)

def visualize(X, y, model):
    plot_decision_boundary(lambda x:predict_values(model,x), X, y)


def plot_decision_boundary(pred_func, X, y):
    """
    This function is responsible for plotting a graph with decision bounderies
    """
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1],alpha=0.8, c=y, cmap=plt.cm.Spectral)
    plt.show()

def main():
   """
   1. Loads the data from voting.csv which is preprocessed after removing excessive colunmns and cleaning up the missing values.
   2. Calls the estimate function which in turn implements K- fold and returns final results of model.
   3. Calls performance_parameters function to compute performance values.
   4. Calls the plot_regression_line the regression line based on performance values.
   """
   filename = 'iris_dataset_final.csv'
   dataset = read_csv(filename)
   for i in range(len(dataset[0]) - 1):
       convert_to_float(dataset, i)
   # convert class column to integers
   convert_to_int(dataset, len(dataset[0]) - 1)
   # normalize input variables
   minmax = dataset_find_minmax(dataset)
   normalize_scale(dataset, minmax)

   number_of_folds = 5
   learning_rate = 0.3
   epoch = 500
   no_of_hidden = 5
   scores = training_algorithm(dataset, number_of_folds, learning_rate, epoch, no_of_hidden)
   print('Scores: %s' % scores)
   print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
   data = pd.read_csv('C://Users//VISHAL//PycharmProjects//NeuralNetwork/iris_dataset_final.csv')
   X = data.iloc[0:150, [0, 1]].values
   y = data.iloc[0:150, 2].values
   # y = np.where(y == 'Iris-setosa', -1, 1)
  # visualize(X,y,initialize_network)
   plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
   plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
   plt.scatter(X[100:150, 0], X[100:150, 1], color='green', marker='^', label='virginica')
   plt.xlabel('petal length')
   plt.ylabel('sepal length')
   plt.legend(loc='upper left')
   plt.show()


if __name__ == '__main__':
   main()
