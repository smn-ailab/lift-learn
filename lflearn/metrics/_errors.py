"""Evaluation metrics for ite predictions."""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def sdr_mse(y: np.ndarray, w: np.ndarray, ite_pred: np.ndarray,
            mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None, gamma: float=0.) -> float:
    """Estimate mean squared error.

    Parameters
    ----------
    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).

    w : array-like, shape = [n_samples]
        The treatment assignment. The values should be binary.

    ite_pred: array-like of shape = (n_samples)
        The predicted values of Individual Treatment Effects.

    ps: array-like, shape = [n_samples]
        The estimated propensity scores.

    mu: array-like, shape = [n_samples]
        The estimated potential outcomes.

    gamma: float, optional (default=0.0)
        The switching hyper-parameter.

    Returns
    -------
    mse: float
        Estimated mean squared error.

    References
    ----------
    [1] Yuta Saito, Hayato Sakata, and Kazuhide Nakata. "Doubly Robust Prediction
        and Evaluation Methods Improve Uplift Modeling for Observaional Data",
        In Proceedings of the SIAM International Conference on Data Mining, 2019.

    """
    # preprocess potential outcome and propensity estimations.
    mu = np.zeros((w.shape[0], 2)) if mu is None else mu
    ps = pd.get_dummies(w).mean(axis=0).values[w] if ps is None else ps[np.arange(w.shape[0]), w]
    # calcurate switch doubly robust outcome.
    direct_estimates = mu[:, 1] - mu[:, 0]
    transformed_outcome = w * (y - mu[:, 1]) / ps[:, 1] - (1 - w) * (y - mu[:, 0]) / ps[:, 0] + direct_estimates
    transformed_outcome[(w == 1) & (ps[:, 1] < gamma)] = direct_estimates[(w == 1) & (ps[:, 1] < gamma)]
    transformed_outcome[(w == 0) & (ps[:, 0] < gamma)] = direct_estimates[(w == 0) & (ps[:, 0] < gamma)]
    # calcurate mse.
    mse = mean_squared_error(transformed_outcome, ite_pred)
    return mse
