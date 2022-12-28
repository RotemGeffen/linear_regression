# imports
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import logging

X, y = load_boston(return_X_y=True)
n, p = X.shape
print(f'p = {p}, n = {n}')


# write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`
class Ols(object):
    def __init__(self):
        self.w = None

    @staticmethod
    def pad(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _fit(self, X, Y):
        # remember pad with 1 before fitting
        X = Ols.pad(X)
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ Y

    def _predict(self, X):
        # return wx
        return Ols.pad(X) @ self.w

    def score(self, X, Y):
        return np.mean((self._predict(X)-Y)**2)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Normalizer():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def predict(self, X):
        # apply normalization
        return (X - self.mean) / self.std


# Implement OrdinaryLinearRegressionGradientDescent model
class OlsGd(Ols):

    def __init__(self, learning_rate=.05,
                 num_iteration=1000,
                 normalize=True,
                 early_stop=True,
                 verbose=True):

        super(OlsGd, self).__init__()
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.early_stop = early_stop
        self.normalize = normalize
        self.normalizer = Normalizer()
        self.verbose = verbose
        self.loss = []

    def loss_function(self, X, Y):
        return np.mean((X @ self.w - Y) ** 2)

    def _fit(self, X, Y, reset=True, track_loss=True):
        # remember to normalize the data before starting
        if self.normalize:
            X = self.normalizer.fit(X).predict(X)
        X = self.pad(X)
        self.w = np.random.randn(X.shape[1])
        stop = False
        i = 1
        prev_loss = 0
        while i <= self.num_iteration:
            curr_loss = self.loss_function(X, Y)
            if track_loss:
                self.loss.append(curr_loss)
            self.w -= self._step(X, Y)
            # Print iteration log if verbose=True
            if self.verbose:
                logging.info(f'Iteration number {i} - the loss is {round(curr_loss, 3)}')
            # Check if early_stop=True and stop the loop in case the condition is met
            if self.early_stop and abs(curr_loss - prev_loss) < 0.0001:
                if self.verbose:
                    logging.info(f'Stopped after {i} iterations due to early stop threshold!')
                break
            prev_loss = curr_loss
            i += 1

    def _predict(self, X):
        # remember to normalize the data before starting
        if self.normalize:
            X = self.normalizer.predict(X)
        return super()._predict(X)

    def _step(self, X, Y):
        # use w update for gradient descent
        return self.learning_rate * (2 / X.shape[0]) * X.T @ (X @ self.w - Y)


# Ridge Linear Regression
class RidgeLs(Ols):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs, self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda

    def _fit(self, X, Y):
        # Closed form of ridge regression
        X = Ols.pad(X)
        identity = self.ridge_lambda * np.eye(X.shape[1])
        self.w = np.linalg.pinv(X.T @ X + identity) @ X.T @ Y

    def score(self, X, Y):
        return super(RidgeLs, self).score(X, Y) + np.sum((self.ridge_lambda * self.w) ** 2) / X.shape[0]


# Ridge Regression - Gradient descent
class RidgeLs_Gd(OlsGd):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs_Gd, self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda

    def loss_function(self, X, Y):
        return np.sum((X @ self.w - Y)**2) + np.sum(self.ridge_lambda * self.w**2)








