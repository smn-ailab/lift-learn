"""Metrics to assess performance on ite prediction task."""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def cab_mse(y: np.ndarray, w: np.ndarray, ite_pred: np.ndarray,
            mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None, gamma: float=0.) -> float:
    """Estimate mean squared error.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = shape = (n_samples)
        Treatment assignment variables.

    ite_pred: array-like of shape = (n_samples)
        Predicted Individual Treatment Effects.

    mu: array-like of shape = (n_samples, 2), optional
        Estimated potential outcomes.

    ps: array-like of shape = (n_samples, 2), optional
        Estimated propensity scores.

    gamma: float, optional
        Switching hyper-parameter.

    Returns
    -------
    mse: float
        Estimated mean squared error.

    """
    mu = np.zeros((w.shape[0], 2)) if mu is None else mu
    ps = pd.get_dummies(w).mean(axis=0).values[w] if ps is None else ps[np.arange(w.shape[0]), w]
    # proxy_outcome = mu[:, 1] - mu[:, 0]
    proxy_outcome = mu[:, 1] - mu[:, 0] \
        + np.minimum(gamma * w / ps, 1) * (y - mu[:, 1]) \
        - np.minimum(gamma * (1 - w) / ps, 1) * (y - mu[:, 0])
    mse = mean_squared_error(proxy_outcome, ite_pred)
    return mse


def ips_mse(y: np.ndarray, w: np.ndarray, ite_pred: np.ndarray, ps: Optional[np.ndarray]=None) -> float:
    """Mean Squared Error Estimator based on Inverse Propensity Score Weighting method.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = shape = (n_samples)
        Treatment assignment indicators.

    ite_pred: array-like of shape = (n_samples)
        Estimated Individual Treatment Effects.

    ps: array-like of shape = (n_samples), optional
        Estimated propensity scores.

    Returns
    -------
    mse: float
        Estimated mean squared error using Inverse Propensity Score Weighting method.

    References
    ----------
    [1] P. Gutierrez, J. Y. Gerardy:  Causal Inference and Uplift Modeling A review of the literature, 2016.

    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray.")
    if not isinstance(w, np.ndarray):
        raise TypeError("w must be a numpy.ndarray.")
    if not isinstance(ite_pred, np.ndarray):
        raise TypeError("ite_pred must be a numpy.ndarray.")
    if ps is None:
        ps = 0.5
    else:
        if not isinstance(ps, np.ndarray):
            raise TypeError("ps must be a numpy.ndarray.")
        assert (np.max(ps) < 1) and (np.min(ps) > 0), "ps must be strictly between 0 and 1."

    transformed_outcome = (w * y / ps[:, 1]) - ((1 - w) * y / ps[:, 0])
    mse = mean_squared_error(transformed_outcome, ite_pred)
    return mse


def sdr_mse(y: np.ndarray, w: np.ndarray, ite_pred: np.ndarray, ps: np.ndarray,
            mu_treat: np.ndarray, mu_control: np.ndarray, gamma: float=0.0) -> float:
    """Mean Squared Error Estimator based on Switch Doubly Robust method.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = shape = (n_samples)
        Treatment assignment indicators.

    ite_pred: array-like of shape = (n_samples)
        Estimated Individual Treatment Effects.

    ps: array-like of shape = (n_samples)
        Estimated propensity scores.

    mu_treat: array-like of shape = (n_samples), optional
        Estimated potential outcome for the treated.

    mu_control: array-like of shape = (n_samples), optional
        Estimated potential outcome for the untreated.

    gamma: float, (default=0,0), optional
        The switching hyper-parameter gamma for SDRM.

    Returns
    -------
    mse: float
        Estimated mean squared error using Doubly Robust method.

    References
    ----------
    [1] Doubly Robust Prediction and Evaluation Methods Improve Uplift Modeling for Observaional Data, 2018.

    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray.")
    if not isinstance(w, np.ndarray):
        raise TypeError("w must be a numpy.ndarray.")
    if not isinstance(ite_pred, np.ndarray):
        raise TypeError("ite_pred must be a numpy.ndarray.")
    if not isinstance(ps, np.ndarray):
        raise TypeError("ps must be a numpy.ndarray.")
    if not isinstance(mu_treat, np.ndarray):
        raise TypeError("mu_treat must be a numpy.ndarray.")
    if not isinstance(mu_control, np.ndarray):
        raise TypeError("mu_control must be a numpy.ndarray.")
    if not isinstance(gamma, float):
        raise TypeError("gamma must be a float.")
    assert (np.max(ps) < 1) and (np.min(ps) > 0), "ps must be strictly between 0 and 1."
    assert (gamma >= 0.0) and (gamma <= 1), "gamma must be between 0 and 1."

    direct_method = mu_treat - mu_control
    sdr_transformed_outcome = w * (y - mu_treat) / ps[:, 1] - (1 - w) * (y - mu_control) / ps[:, 0] + (mu_treat - mu_control)
    sdr_transformed_outcome[(w == 1) & (ps[:, 1] < gamma)] = direct_method[(w == 1) & (ps[:, 1] < gamma)]
    sdr_transformed_outcome[(w == 0) & (ps[:, 0] < gamma)] = direct_method[(w == 0) & (ps[:, 0] < gamma)]
    mse = mean_squared_error(sdr_transformed_outcome, ite_pred)
    return mse
