
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

#read dataset using read csv function
data = pd.read_csv('C://Users//VISHAL//PycharmProjects//NeuralNetwork/wine_data.csv')
data.head()
print(data.keys())


X = data.iloc[0:178, 0:12].values
y = data.iloc[0:178, 13].values

# split the data into traning and testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# Sci-kit learn svm model with linear kernel method
clf = GridSearchCV(svm.SVC(kernel = 'linear'), cv=5,
                           param_grid = {'C': [1, 10, 100, 1000]})

clf.fit(x_train, y_train)
print(clf.best_params_)
y_pred = clf.predict(x_test)
print("Linear : " + str(clf.score(x_test, y_test)))

# Plot the graph for linear svm kernel
def plot_decision_regions(X, y, classifier, resolution = 0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    print(np.unique(y))

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

#plot_decision_regions(X, y, classifier=clf)
#plt.xlabel('x1')
#plt.ylabel('x2')
#plt.legend(loc='upper left')
#plt.show()


# Sci-kit learn svm model with polynomial kernel method
clf = GridSearchCV(svm.SVC(kernel='poly', degree=3), cv=5,
                param_grid = {'C': [1, 10, 100, 1000]})

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(clf.best_params_)
print("Poly:" + str(clf.score(x_test, y_test)))

# Plot the graph for svm polynomial kernel
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    print(np.unique(y))

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

#plot_decision_regions(X, y, classifier=clf)
#plt.xlabel('x1')
#plt.ylabel('x2')
#plt.legend(loc='upper left')
#plt.show()

# Sci-kit learn svm model with gaussian kernel method
clf = GridSearchCV(svm.SVC(kernel='rbf'), cv=5,
               param_grid={'C': [1, 10, 100, 1000],
                         'gamma':[0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]})

clf.fit(x_train, y_train)
print(clf.best_params_)
y_pred = clf.predict(x_test)
print("rbf: " + str(clf.score(x_test, y_test)))

# Plot the graph for svm gaussian kernel
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # print(X[:,0])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    print(np.unique(y))

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        # print(idx)
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

#plot_decision_regions(X, y, classifier=clf)
#plt.xlabel('x1')
#plt.ylabel('x2')
#plt.legend(loc='upper left')
#plt.show()