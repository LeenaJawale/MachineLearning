from csv import reader
from random import randrange
from math import exp


def training_algo(dataset, number_of_folds, learning_rate, epoch):
    """
    This function is responsible for:
       1. Creating 5 folds in which row values are added based on an index of data set which is generated randomly.
       2. For splitting the data into two sets training data set and testing data set.
       3. Calling the function 'algorithm' for training the algorithm.
       4. Calling the accuracy function for figuring the actual accuracy results.
    """
    k_folds = list()
    size_of_fold = int(len(dataset) / number_of_folds)
    for k in range(number_of_folds):
        fold = list()
        while len(fold) < size_of_fold:
            index = randrange(len(dataset))
            fold.append(dataset.pop(index))
        k_folds.append(fold)
    final_results = list()
    for fold in k_folds:
        training_data = list(k_folds)
        training_data.remove(fold)
        training_data = sum(training_data, [])
        testing_data = list()
        for line in fold:
            line = list(line)
            testing_data.append(line)
            line[-1] = None
        predicted_values = algorithm(training_data, testing_data,learning_rate, epoch)
        actual_values = [row[-1] for row in fold]
        accuracy = compute_accuracy(actual_values, predicted_values)
        final_results.append(accuracy)
    return final_results


def compute_accuracy(actual_values, predicted_values):
    """
    This function is responsible for computing the accuracy of predicted values against actual values.
    :param actual_values: Actual Y values
    :param predicted_values: Predicted Y values
    :return: Accuracy percentage for a given fold.
    """
    accurate = 0
    for j in range(len(actual_values)):
        if actual_values[j] == predicted_values[j]:
            accurate += 1
    accuracy = accurate / float(len(actual_values)) * 100.0
    return accuracy


def algorithm(training_data, testing_data, learning_rate, epoch):
    """
    This function is responsible for:
    1. Calling stochastic_gradient_descent function for computing regression coefficients.
    2. Using the coefficient values, calculate the predictions.
    :param training_data: Training data set.
    :param testing_data: Testing data set.
    :param learning_rate: Limit the amount each coefficient is corrected each time it is updated.
    :param epoch: The number of times to run through training data to update coefficients.
    :return: A list of predicted Y values
    """
    predictions = list()
    coef = stochastic_gradient_descent(training_data, learning_rate, epoch)
    for line in testing_data:
        predicted_output = round(predict_coeff(line, coef))
        predictions.append(predicted_output)
    return predictions


def stochastic_gradient_descent(training_data, learning_rate, epoch):
    """
    This function is responsible for:
    1. updating coefficient for each row in the training set each epoch, by looping over each epoch, over each row and update over each coefficient.
    :param training_data: Training data set
    :param learning_rate: Limit the amount each coefficient is corrected each time it is updated.
    :param epoch: The number of times to run through training data to update coefficients.
    :return: A list of coefficients having intercept and b values.
    """
    coefficient = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(epoch):
        for line in training_data:
            predicted_output = predict_coeff(line, coefficient)
            error = line[-1] - predicted_output
            coefficient[0] = coefficient[0] + learning_rate * error * predicted_output * (1.0 - predicted_output)
            for i in range(len(line) - 1):
                coefficient[i + 1] = coefficient[i + 1] + learning_rate * error * predicted_output * (1.0 - predicted_output) * line[i]
    return coefficient


def predict_coeff(line, coefficients):
    """
    This function is responsible for making predictions with coefficients.
    :param line: Represents each row of training or testing data set.
    :param coefficients: Value of sigmoid
    """
    predicted_value = coefficients[0]
    for i in range(len(line) - 1):
        predicted_value = predicted_value + coefficients[i + 1] * line[i]
    return 1.0 / (1.0 + exp(-predicted_value))


def create_list(source):
    """
    This function is responsible for loading the data in a list.
    :param source: Source data set
    :return: Loaded data set in a list
    """
    data_set = list()
    with open(source, 'r') as file:
        read_csv = reader(file)
        for line in read_csv:
            if not line:
                continue
            data_set.append(line)
    return data_set


def convert_to_float(data, j):
    """
    This function is responsible for converting string columns to float.
    """
    for line in data:
        line[j] = float(line[j].strip())


def normalize_scale(data):
    """
    This function is responsible for normalizing/ scaling the data set to the range of 0-1
    """
    max_min = list()
    for k in range(len(data[0])):
        column = [line[k] for line in data]
        maximum = max(column)
        minimum = min(column)
        max_min.append([minimum, maximum])
    for line in data:
        for i in range(len(line)):
            line[i] = (line[i] - max_min[i][0]) / (max_min[i][1] - max_min[i][0])


def main():
   """
   1. Loads the data from voting.csv which is preprocessed after removing excessive colunmns and cleaning up the missing values.
   2. Calls the estimate function which in turn implements K- fold and returns final results of model.
   3. Calls performance_parameters function to compute performance values.
   4. Calls the plot_regression_line the regression line based on performance values.
   """
   number_of_folds = 5
   learning_rate = 0.1
   epoch = 100
   source = 'voting_cleaned_data.csv'
   dataset = create_list(source)
   for j in range(len(dataset[0])):
    convert_to_float(dataset, j)
   normalize_scale(dataset)
   results = training_algo(dataset,number_of_folds, learning_rate, epoch)
   print('Accuracy Results: %s' % results)
   print('Overall Accuracy: %.3f%%' % (sum(results) / float(len(results))))

if __name__ == '__main__':
   main()