import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator


class ColumnExtractor(BaseEstimator):
    """
    Simple column extractor
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns:
            return X[self.columns]
        else:
            return X


def crps(true, pred):
    """
    Calculation of CRPS.

    :param true: true values (labels)
    :param pred: predicted values
    """
    return np.sum(np.square(true - pred)) / true.size


def real_to_cdf(y, sigma=1e-10):
    """
    Utility function for creating CDF from real number and sigma (uncertainty measure).

    :param y: array of real values
    :param sigma: uncertainty measure. The higher sigma, the more imprecise the prediction is, and vice versa.
    Default value for sigma is 1e-10 to produce step function if needed.
    """
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf
