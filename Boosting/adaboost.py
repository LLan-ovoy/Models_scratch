import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    X = []
    Y = []
    with open(filename) as f:
        for line in f:
            data = line.replace('\n','').split(',')
            X.append(data[:-1])
            Y.append((int(data[-1])-0.5)/0.5)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights

    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = []
    N, _ = X.shape
    d = np.ones(N) / N
    for m in range(num_iter):
        stump = DecisionTreeClassifier(max_depth = max_depth).fit(X,y,d)
        y_hat = stump.predict(X)

        err_m = np.sum(d * abs(y_hat-y)/2)/np.sum(d)
        if err_m == 0:
            tree_w = 1
        else:
            tree_w = np.log((1-err_m)/err_m)

        trees.append(stump)
        trees_weights.append(tree_w)

        d = d * np.exp(abs(y_hat-y)/2*tree_w)
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    N, _ =  X.shape
    y = np.zeros(N)
    pred_trees = np.array([tree.predict(X) for tree in trees])
    trees_weights_m = np.array([trees_weights for _ in range(pred_trees.shape[1])])
    y = np.sign(np.sum(pred_trees.T * trees_weights_m, axis=1))
    return y
