import numpy as np
from statistics import mode
from sklearn import metrics


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)

    def leaf(self, x_test):
        if x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)
        return self.rchild.leaf(x_test)



class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        self.y = y

    def predict(self, x_test):
        return self.prediction

    def leaf(self, x_test):
        return self



def gini(y):
    "Return the gini impurity score for values in y"
    count = np.unique(y, return_counts=True)
    p = count[1]/len(y)
    score = 1 - np.sum(p**2)
    return score


class DecisionTree621:
    def __init__(self, max_features, min_samples_leaf=1, loss=None):
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini


    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def bestsplit(self, X, y):
        "find the best split"
        best = (-1, -1, self.loss(y))
        var = np.random.choice(a=range(len(X[0])), size=round(self.max_features*len(X[0])))
        for col in var:
            candidates = np.random.choice(a=X[:,col], size=11)
            for split in candidates:
                yl = y[X[:,col]<=split]
                yr = y[X[:,col]>split]
                if len(yl) < self.min_samples_leaf or len(yr) < self.min_samples_leaf:
                    continue
                l = (len(yl) * self.loss(yl) + len(yr) * self.loss(yr))/len(y)
                if l == 0:
                    return col, split
                elif l < best[2]:
                    best = (col, split, l)
        return best[0], best[1]


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) < self.min_samples_leaf:
            return self.create_leaf(y)
        col, split = self.bestsplit(X, y)
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:,col]<=split], y[X[:,col]<=split])
        rchild = self.fit_(X[X[:,col]>split], y[X[:,col]>split])
        return DecisionNode(col, split, lchild, rchild)


    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        y_pred = [self.root.predict(record) for record in X_test]
        return np.array(y_pred)

    def leaf(self, x_test):
        return self.root.leaf(x_test)


class RegressionTree621(DecisionTree621):
    def __init__(self, max_features, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)
        self.max_features = max_features
        self.oob_idxs = np.nan


    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_hat = self.predict(X_test)
        r_2 = metrics.r2_score(y_test, y_hat)
        return r_2


    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, max_features, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)
        self.max_features = max_features
        self.oob_idxs = np.nan


    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_hat = self.predict(X_test)
        accuracy_score = metrics.accuracy_score(y_test, y_hat)
        return accuracy_score


    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.argmax(np.bincount(y)))
