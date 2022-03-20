import numpy as np
from dtree import *
from collections import defaultdict
from sklearn import metrics
import scipy


class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        for i in range(self.n_estimators):
            bs_idxs = np.random.randint(0, X.shape[0], size=X.shape[0])
            oob_idxs = np.setdiff1d(np.arange(X.shape[0]), bs_idxs, assume_unique=True)
            X_bs = X[bs_idxs]
            y_bs = y[bs_idxs]
            self.trees[i].fit(X_bs, y_bs)
            self.trees[i].oob_idxs = oob_idxs

        if self.oob_score:
            self.oob_score_ = self.compute_obb_score(X, y)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = [RegressionTree621(max_features=max_features, \
                                        min_samples_leaf=min_samples_leaf) \
                                        for _ in range(n_estimators)]

    def compute_obb_score(self, X, y):
        oob_c = np.zeros((X.shape[0], self.n_estimators))
        oob_p = np.zeros((X.shape[0], self.n_estimators))
        for i, dtree in enumerate(self.trees):
            for idx in dtree.oob_idxs:
                leaf = dtree.leaf(X[idx])
                oob_c[idx, i] = leaf.n
                oob_p[idx, i] = leaf.n * leaf.prediction
        oob_counts = np.sum(oob_c, axis=1)
        nonzero_idx = np.where(oob_counts > 0)
        oob_pred = np.sum(oob_p,axis=1)
        oob_avg_preds = oob_pred[nonzero_idx]/oob_counts[nonzero_idx]
        return metrics.r2_score(y[nonzero_idx], oob_avg_preds)

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        y_pred = []
        for x in X_test:
            leaves = [dtree.leaf(x).y for dtree in self.trees]
            lenleaves = [dtree.leaf(x).n for dtree in self.trees]
            nobs = np.sum(lenleaves)
            ysum = np.sum([np.sum(leaf) for leaf in leaves])
            y_pred.append(ysum/nobs)
        return np.array(y_pred)

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return metrics.r2_score(y_test, self.predict(X_test))


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)

        self.trees = [ClassifierTree621(max_features=max_features, \
                                        min_samples_leaf=min_samples_leaf) \
                                        for _ in range(n_estimators)]

    def compute_obb_score(self, X, y):
        oob_counts = np.zeros(X.shape[0])
        oob_votes = np.zeros(X.shape[0])
        oob_pred = np.zeros((X.shape[0], 100))
        for dtree in self.trees:
            oob_idxs = dtree.oob_idxs
            leafsizes = [dtree.leaf(x).n for x in X[oob_idxs]]
            tpred = dtree.predict(X[oob_idxs])
            oob_pred[oob_idxs, tpred] += leafsizes
            oob_counts[oob_idxs] += 1
        nonzero_idx = np.where(oob_counts > 0)
        oob_votes[nonzero_idx] = np.argmax(oob_pred[nonzero_idx],1)
        return metrics.accuracy_score(y[nonzero_idx], oob_votes[nonzero_idx])

    def predict(self, X_test) -> np.ndarray:
        y_pred = []
        for x in X_test:
            counts = defaultdict(int)
            for dtree in self.trees:
                leaf = dtree.leaf(x).y
                for y in leaf:
                    counts[y] += 1
            y_pred.append(max(counts, key=counts.get))
        return y_pred

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return metrics.accuracy_score(y_test, self.predict(X_test))
