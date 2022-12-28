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


model_ols = Ols()
model_ols._fit(X, y)
training_mse = model_ols.score(X, y)
print(f'The training MSE is: {training_mse}')






