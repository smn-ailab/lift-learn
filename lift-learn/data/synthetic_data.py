"""This Module contains a class to generate synthetic data for uplift modeling."""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class DataGenerator():
    """Base Class to assess performance of meta learning methods and evaluation metrics for offline uplift modeling."""

    def __init__(self, num_data: int, num_features: int, num_trts: int, scenario: int,
                 irrelevant_feature_type: Optional[str]=None, p: Optional[List]=None, ate: Optional[List]=None,) -> None:
        """Initialize Class.

        Parameters
        ----------
        num_data: int
            Number of data for a simulation.

        num_features: int
            Number of features for an unit.

        num_trts: int
            Number of treatments.

        scenario: int, in [1, 2, 3, 4, 5, 6, 7, 8]
            A data generating scenario.

        """
        self.num_data = num_data
        self.num_features = num_features
        self.num_trts = num_trts
        self.scenario = scenario
        self.irrelevant_feature_type = irrelevant_feature_type
        self.p = p
        self.ate = np.zeros(self.num_trts) if ate is None else ate

    def generate_data(self, test_size: float=0.5, noise: float=1.0, rct: bool=True,
                      random_state: int=0, seed: int=0) -> None:
        """Generate synthetic data.

        Parameters
        ----------
        test_size: float (default=0.5)
            Represent the proportion of the dataset to include in the test split,
            and should be between 0.0 and 1.0.

        p : 1-D array-like, optional
            The probabilities associated with each entry in a.
            If not given the sample assumes a uniform distribution over all entries in a.

        random_state: int (default=0)
            The seed used by the train/test split.

        seed: int (default=0)
            The seed used by the random number generator.

        """
        X = [f"x{i + 1}" for i in np.arange(self.num_features)]
        np.random.seed(seed)
        if self.irrelevant_feature_type is None:
            data = np.vstack([np.random.normal(loc=0, scale=1, size=self.num_data), np.random.binomial(n=1, p=0.5, size=self.num_data)]
                             for i in np.arange(int(self.num_features / 2))).T
        else:
            relevant_data = np.vstack([np.random.normal(loc=0, scale=1, size=self.num_data),
                                       np.random.binomial(n=1, p=0.5, size=self.num_data)] for i in np.arange(5)).T

            # if self.irrelevant_feature_type == "gaussian":
            #    irrelevant_data = np.vstack([np.random.normal(loc=0, scale=1, size=self.num_data)] for i in np.arange(self.num_features - 10)).T

            # elif self.irrelevant_feature_type == "binomial":
            irrelevant_data = np.vstack([np.random.choice(self.irrelevant_feature_type, size=self.num_data)] for i in np.arange(self.num_features - 10)).T

            data = np.hstack([relevant_data, irrelevant_data])

        self.df = DataFrame(data, columns=X)

        # generate synthetic data.
        self.df["mu"] = self._mu(data)
        ite = self._treatment_effect(data)
        for i in np.arange(self.num_trts):
            self.df[f"mu{i}"] = self.df.mu + ite[:, i] + self.ate[i]
        if rct:
            self.df["w"] = np.random.choice(np.arange(self.num_trts), p=self.p, size=self.num_data)
            self.df["y_obs"] = self._y_obs(self.df.mu.values, ite, self.df.w.values, noise)
            self.df.reindex(X + [f"mu{i}" for i in np.arange(self.num_trts)] + ["y_obs", "w"], axis=1)
        else:
            ps = softmax(self.df[[f"mu{i}" for i in np.arange(self.num_trts)]].values)
            for i in np.arange(self.num_trts):
                self.df[f"ps{i}"] = ps[:, i]
            self.df["w"] = np.array([np.random.choice(np.arange(self.num_trts), size=1, p=ps[i]) for i in np.arange(self.num_data)])
            self.df["y_obs"] = self._y_obs(self.df.mu.values, ite, self.df.w.values, noise)
            self.df.reindex(X + [f"mu{i}" for i in np.arange(self.num_trts)] + ["y_obs", "w"] +
                            [f"ps{i}" for i in np.arange(self.num_trts)], axis=1)

    def _mu(self, X: np.ndarray) -> np.ndarray:
        """Generate expected outcome values given w = 0.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Input samples.

        Returns
        -------
        mu  : array-like of shape = (n_samples)
            Expected outcome values given w = 0.

        """
        key = 0
        if self.scenario == 1:
            mu = self._f8(X, key)

        elif self.scenario == 2:
            mu = self._f5(X, key)

        elif self.scenario == 3:
            mu = self._f4(X, key)

        elif self.scenario == 4:
            mu = self._f7(X, key)

        elif self.scenario == 5:
            mu = self._f3(X, key)

        elif self.scenario == 6:
            mu = self._f1(X)

        elif self.scenario == 7:
            mu = self._f2(X, key)

        elif self.scenario == 8:
            mu = self._f6(X, key)

        return mu

    def _treatment_effect(self, X: np.ndarray) -> np.ndarray:
        """Generate Individual Treatment Effects.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Input samples.

        Returns
        -------
        ite  : array-like of shape = (n_samples)
            Individual Treatment Effects.

        """
        if self.scenario == 1:
            if self.num_trts == 2:
                ite = np.c_[- self._f1(X), self._f1(X)]

        elif self.scenario == 2:
            if self.num_trts == 2:
                ite = np.c_[- self._f2(X, 0), self._f2(X, 0)]
            elif self.num_trts == 4:
                ite = np.c_[- self._f2(X, 0), self._f2(X, 0), - self._f2(X, 1), self._f2(X, 1)]
            elif self.num_trts == 6:
                ite = np.c_[- self._f2(X, 0), self._f2(X, 0), - self._f2(X, 1), self._f2(X, 1), - self._f2(X, 2), self._f2(X, 2)]

        elif self.scenario == 3:
            if self.num_trts == 2:
                ite = np.c_[- self._f3(X, 0), self._f3(X, 0)]
            elif self.num_trts == 4:
                ite = np.c_[- self._f3(X, 0), self._f3(X, 0), - self._f3(X, 1), self._f3(X, 1)]
            elif self.num_trts == 6:
                ite = np.c_[- self._f3(X, 0), self._f3(X, 0), - self._f3(X, 1), self._f3(X, 1), - self._f3(X, 2), self._f3(X, 2)]

        elif self.scenario == 4:
            if self.num_trts == 2:
                ite = np.c_[- self._f4(X, 0), self._f4(X, 0)]
            elif self.num_trts == 4:
                ite = np.c_[- self._f4(X, 0), self._f4(X, 0), - self._f4(X, 1), self._f4(X, 1)]
            elif self.num_trts == 6:
                ite = np.c_[- self._f4(X, 0), self._f4(X, 0), - self._f4(X, 1), self._f4(X, 1), - self._f4(X, 2), self._f4(X, 2)]

        elif self.scenario == 5:
            if self.num_trts == 2:
                ite = np.c_[- self._f5(X, 0), self._f5(X, 0)]
            elif self.num_trts == 4:
                ite = np.c_[- self._f5(X, 0), self._f5(X, 0), - self._f5(X, 1), self._f5(X, 1)]
            elif self.num_trts == 6:
                ite = np.c_[- self._f5(X, 0), self._f5(X, 0), - self._f5(X, 1), self._f5(X, 1), - self._f5(X, 2), self._f5(X, 2)]

        elif self.scenario == 6:
            if self.num_trts == 2:
                ite = np.c_[- self._f6(X, 0), self._f6(X, 0)]
            elif self.num_trts == 4:
                ite = np.c_[- self._f6(X, 0), self._f6(X, 0), - self._f6(X, 1), self._f6(X, 1)]
            elif self.num_trts == 6:
                ite = np.c_[- self._f6(X, 0), self._f6(X, 0), - self._f6(X, 1), self._f6(X, 1), - self._f6(X, 2), self._f6(X, 2)]

        elif self.scenario == 7:
            if self.num_trts == 2:
                ite = np.c_[- self._f7(X, 0), self._f7(X, 0)]
            elif self.num_trts == 4:
                ite = np.c_[- self._f7(X, 0), self._f7(X, 0), - self._f7(X, 1), self._f7(X, 1)]
            elif self.num_trts == 6:
                ite = np.c_[- self._f7(X, 0), self._f7(X, 0), - self._f7(X, 1), self._f7(X, 1), - self._f7(X, 2), self._f7(X, 2)]

        elif self.scenario == 8:
            if self.num_trts == 2:
                ite = np.c_[- self._f8(X, 0), self._f8(X, 0)]
            elif self.num_trts == 4:
                ite = np.c_[- self._f8(X, 0), self._f8(X, 0), - self._f8(X, 1), self._f8(X, 1)]
            elif self.num_trts == 6:
                ite = np.c_[- self._f8(X, 0), self._f8(X, 0), - self._f8(X, 1), self._f8(X, 1), - self._f8(X, 2), self._f8(X, 2)]

        return ite / 2

    def _y_obs(self, mu: np.ndarray, ite: np.ndarray, w: np.ndarray, noise: float, seed: int=0) -> np.ndarray:
        """Generate observed outcome values.

        Parameters
        ----------
        mu_treat: array-like of shape = (n_samples)
            Estimated target values given w = 1.

        mu_control: array-like of shape = (n_samples)
            Estimated target values given w = 0.

        w: array-like of shape = shape = (n_samples)
            Treatment assignment indicators.

        noise: int
            Standard deviation for gaussian noise on observed outcome values.

        seed: int (default=0)
            The seed used by the random number generator.

        Returns
        -------
        y_obs: array-like of shape = (n_samples)

        """
        np.random.seed(seed)
        w = pd.get_dummies(w).values
        mus = np.expand_dims(mu, axis=1) + ite
        y_obs = np.sum(mus * w, axis=1) + np.random.normal(0, noise, size=w.shape[0])

        return y_obs

    def _f1(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])

    def _f2(self, X: np.ndarray, key: int) -> np.ndarray:
        if key == 0:
            result = 6 * self._indicator(X[:, 0], 1) - 6 * norm.cdf(-1)
        elif key == 1:
            result = 6 * self._indicator(X[:, 2], 1) - 6 * norm.cdf(-1)
        elif key == 2:
            result = 6 * self._indicator(X[:, 4], 1) - 6 * norm.cdf(-1)

        return result

    def _f3(self, X: np.ndarray, key: int) -> np.ndarray:
        if key == 0:
            result = 5 * X[:, 0]
        elif key == 1:
            result = 2 * X[:, 2] - 1
        elif key == 2:
            result = 4 * X[:, 4] - 2

        return result

    def _f4(self, X: np.ndarray, key: int) -> np.ndarray:
        if key == 0:
            result = X[:, 1] * X[:, 3] * X[:, 5] + 2 * X[:, 1] * X[:, 3] * (1 - X[:, 5]) + 3 * X[:, 1] * (1 - X[:, 3]) * X[:, 5] \
                + 4 * X[:, 1] * (1 - X[:, 3]) * (1 - X[:, 5]) + 5 * (1 - X[:, 1]) * X[:, 3] * X[:, 5] + 6 * (1 - X[:, 1]) * X[:, 3] * (1 - X[:, 5]) \
                + 7 * (1 - X[:, 1]) * (1 - X[:, 3]) * X[:, 5] + 8 * (1 - X[:, 1]) * (1 - X[:, 3]) * (1 - X[:, 5]) - 4.5
        elif key == 1:
            result = 3 * X[:, 1] * X[:, 3] * X[:, 5] - 2 * X[:, 1] * X[:, 3] * (1 - X[:, 5]) + 5 * X[:, 1] * (1 - X[:, 3]) * X[:, 5] \
                - 3 * X[:, 1] * (1 - X[:, 3]) * (1 - X[:, 5]) - (1 - X[:, 1]) * X[:, 3] * X[:, 5] + 7 * (1 - X[:, 1]) * X[:, 3] * (1 - X[:, 5]) \
                + (1 - X[:, 1]) * (1 - X[:, 3]) * X[:, 5] - 4 * (1 - X[:, 1]) * (1 - X[:, 3]) * (1 - X[:, 5]) - 0.5
        elif key == 2:
            result = 2 * X[:, 1] * X[:, 3] * X[:, 5] + 4 * X[:, 1] * X[:, 3] * (1 - X[:, 5]) - 3 * X[:, 1] * (1 - X[:, 3]) * X[:, 5] \
                + 2 * X[:, 1] * (1 - X[:, 3]) * (1 - X[:, 5]) + 2 * (1 - X[:, 1]) * X[:, 3] * X[:, 5] + (1 - X[:, 1]) * X[:, 3] * (1 - X[:, 5]) \
                - 3 * (1 - X[:, 1]) * (1 - X[:, 3]) * X[:, 5] - (1 - X[:, 1]) * (1 - X[:, 3]) * (1 - X[:, 5]) + 0.5

        return result

    def _f5(self, X: np.ndarray, key: int) -> np.ndarray:
        if key == 0:
            result = X[:, 0] + X[:, 2] + X[:, 4] + X[:, 6] + X[:, 7] + X[:, 8] - 0.5
        elif key == 1:
            result = X[:, 0] - X[:, 2] + X[:, 4] - X[:, 6] + X[:, 5] - X[:, 8]
        elif key == 2:
            result = X[:, 0] - X[:, 2] + X[:, 4] - X[:, 6] + X[:, 3] - X[:, 8]

        return result

    def _f6(self, X: np.ndarray, key: int) -> np.ndarray:
        if key == 0:
            result = 4 * self._indicator(X[:, 0], 1) * self._indicator(X[:, 2], 0) + 4 * self._indicator(X[:, 4], 1) * self._indicator(X[:, 6], 0) \
                + 2 * X[:, 7] * X[:, 8] - 4 * norm.cdf(-1)
        elif key == 1:
            result = 4 * self._indicator(X[:, 2], 1) * self._indicator(X[:, 4], 0) + 4 * self._indicator(X[:, 6], 1) * self._indicator(X[:, 8], 0) \
                + 2 * X[:, 4] * X[:, 5] - 8 * norm.cdf(-1)
        elif key == 2:
            result = 4 * self._indicator(X[:, 0], 1) * self._indicator(X[:, 8], 0) + 4 * self._indicator(X[:, 2], 1) * self._indicator(X[:, 6], 0) \
                + 2 * X[:, 2] * X[:, 3] - 8 * norm.cdf(-1)

        return result

    def _f7(self, X: np.ndarray, key: int) -> np.ndarray:
        if key == 0:
            result = (X[:, 0] ** 2 + X[:, 1] + X[:, 2] ** 2 + X[:, 3] + X[:, 4] ** 2 + X[:, 5] + X[:, 6] ** 2 + X[:, 7] + X[:, 8] ** 2 - 7) / np.sqrt(2)
        elif key == 1:
            result = (2 * X[:, 1] + X[:, 2] ** 2 + 2 * X[:, 3] + 2 * X[:, 5] + X[:, 6] ** 2 + 2 * X[:, 7] - 5.5) / np.sqrt(2)
        elif key == 2:
            result = (X[:, 0] ** 2 + 4 * X[:, 1] + 4 * X[:, 3] + X[:, 4] ** 2 + 4 * X[:, 5] + 4 * X[:, 7] + X[:, 8] ** 2 - 11) / np.sqrt(2)

        return result

    def _f8(self, X: np.ndarray, key: int) -> np.ndarray:
        result = (self._f4(X, key) + self._f5(X, key)) / np.sqrt(2)
        return result

    def _indicator(self, X: np.ndarray, threshold: float) -> float:
        """Calculate indicator function."""
        indicator = np.zeros(X.shape[0])
        indicator[np.where(X > threshold)] = 1
        return indicator


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax Function."""
    exp_x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))

    return exp_x / np.expand_dims(np.sum(exp_x, axis=1), axis=1)


def relu(x: np.ndarray) -> float:
    """Relu function."""
    return np.maximum(0, x)
