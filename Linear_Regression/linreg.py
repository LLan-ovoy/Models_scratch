import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def normalize(X): # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s

def loss_gradient(X, y, B, lmbda):
    result = - np.dot(X.T, (y - np.dot(X, B)))
    return result

def loss_ridge(X, y, B, lmbda):
    result = np.dot((y - np.dot(X, B)).T, (y - np.dot(X, B))) + lmbda * np.dot(B.T, B)
    return result

def loss_gradient_ridge(X, y, B, lmbda):
    result = - np.dot(X.T, (y - np.dot(X, B))) + lmbda * B
    return result

def sigmoid(z):
    result = np.exp(z)/(np.exp(z) + 1)
    return result

def log_likelihood(X, y, B,lmbda):
    result = - np.sum(np.multiply(y, np.dot(X,B)) - np.log(1 + np.exp(np.dot(X, B))))
    return result

def log_likelihood_gradient(X, y, B, lmbda):
    result = - np.dot(X.T, (y - sigmoid(np.dot(X, B))))
    return result

# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood(X, y, B, lmbda):
    result = - np.sum(np.multiply(y, np.dot(X,B)) - np.log(1 + np.exp(np.dot(X, B)))) + lmbda * np.sum(abs(B))
    return result

# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood_gradient(X, y, B, lmbda):
    """
    Must compute \beta_0 differently from \beta_i for i=1..p.
    \beta_0 is just the usual log-likelihood gradient
    # See https://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
    # See https://stackoverflow.com/questions/38853370/matlab-regularized-logistic-regression-how-to-compute-gradient
    """
    err = y - sigmoid(np.dot(X, B))
    B0_g = np.mean(err)
    r = lmbda * np.sign(B)
    r[0] = 0
    result = - (np.dot(X.T, err) - r)
    return result

def minimize(X, y, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=1e-9):
    "Here are various bits and pieces you might want"
    # X = normalize(X)

    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    if addB0:  # add column of 1s to X
        B0 = 1
        X = np.insert(X, 0, B0, axis=1)
        p = p + 1

    B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)

    prev_B = B
    eps = 1e-5 # prevent division by 0

    h = np.zeros(shape=(p, 1))
    for i in range(max_iter):
        l_g_cur = loss_gradient(X, y, B, lmbda)
        if np.linalg.norm(l_g_cur) < precision:
            break
        h += l_g_cur * l_g_cur
        B = B - eta * l_g_cur/(h**0.5 + eps)
    return B


class LinearRegression621: # REQUIRED
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class LogisticRegression621: # REQUIRED
    "Use the above class as a guide."
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        """
        Compute the probability that the target is 1. Basically do
        the usual linear regression and then pass through a sigmoid.
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return sigmoid(np.dot(X, self.B))

    def predict(self, X):
        """
        Call self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        prob = self.predict_proba(X)
        pred = np.where(prob>0.5, 1, 0)
        return pred

    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class RidgeRegression621: # REQUIRED
    "Use the above class as a guide."
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        beta_k = minimize(X, y,
                          loss_gradient_ridge,
                          self.eta,
                          self.lmbda,
                          self.max_iter,
                          addB0 = False)
        beta_0 = np.mean(y)
        self.B = np.insert(beta_k, 0, beta_0, 0)


# NOT REQUIRED but to try to implement for fun
class LassoLogistic621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        """
        Compute the probability that the target is 1. Basically do
        the usual linear regression and then pass through a sigmoid.
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return sigmoid(np.dot(X, self.B))

    def predict(self, X):
        """
        Call self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        prob = self.predict_proba(X)
        pred = np.where(prob>0.5, 1, 0)
        return pred

    def fit(self, X, y):
        # beta_k = minimize(X, y,
        #                   L1_log_likelihood_gradient,
        #                   self.eta,
        #                   self.lmbda,
        #                   self.max_iter,
        #                   addB0 = False)
        # beta_0s = minimize(X, y,
        #                   log_likelihood_gradient,
        #                   self.eta,
        #                   self.lmbda,
        #                   self.max_iter,
        #                   addB0 = True)
        # beta_0 = beta_0s[0]
        # self.B = np.insert(beta_k, 0, beta_0, 0)
        self.B = minimize(X, y,
                          L1_log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter,
                          addB0 = True)
