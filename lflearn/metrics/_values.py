"""Metrics to assess performance on ite prediction task."""
from typing import Optional

import numpy as np
import pandas as pd


def expected_response(y: np.ndarray, w: np.ndarray, policy: np.ndarray,
                      mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None) -> float:
    """Estimate expected response.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = shape = (n_samples)
        Treatment assignment variables.

    policy: array-like of shape = (n_samples)
        Estimated treatment policy.

    mu: array-like of shape = (n_samples, n_trts), optional
        Estimated potential outcomes.

    ps: array-like of shape = (n_samples, n_trts), optional
        Estimated propensity scores.

    Returns
    -------
    expected_response: float
        Estimated expected_response.

    """
    mu = np.zeros((w.shape[0], np.unique(w).shape[0])) if mu is None else mu
    ps = pd.get_dummies(w).mean(axis=0).values if ps is None else ps
    indicator = np.array(w == policy, dtype=int)
    expected_response = np.mean(mu[np.arange(w.shape[0]), policy]
                                + (y - mu[np.arange(w.shape[0]), policy]) * indicator / ps[w])

    return expected_response


def ips_value(y: np.ndarray, w: np.ndarray, policy: np.ndarray, ps: Optional[np.ndarray]=None) -> float:
    """Decision Value Estimator based on Inverse Propensity Score Weighting method.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = shape = (n_samples)
        Treatment assignment indicators.

    policy: array-like of shape = (n_samples)
        Estimated decision model.

    ps: array-like of shape = (n_samples), optional
        Estimated propensity scores.

    Returns
    -------
    decision_value: float
        Estimated decision value using Inverse Propensity Score Weighting method.

    References
    ----------
    [1] Y. Zhao, X. Fang, D. S. Levi: Uplift modeling with multiple treatments and general response types, 2017.
    [2] A. Schuler, M. Baiocchi, R. Tibshirani, N. Shah: A comparison of methods for model selection when estimating individual treatment effects, 2018.

    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray.")
    if not isinstance(w, np.ndarray):
        raise TypeError("w must be a numpy.ndarray.")
    if not isinstance(policy, np.ndarray):
        raise TypeError("policy must be a numpy.ndarray.")
    if ps is None:
        trts_probs = pd.get_dummies(w).mean(axis=0).values
        ps = np.ones((w.shape[0], np.unique(w).shape[0])) * np.expand_dims(trts_probs, axis=0)
    else:
        assert (np.max(ps) < 1) and (np.min(ps) > 0), "ps must be strictly between 0 and 1."

    treatment_matrix = pd.get_dummies(w).values
    if np.unique(policy).shape[0] == np.unique(w).shape[0]:
        policy = pd.get_dummies(policy).values
    else:
        diff = np.setdiff1d(np.unique(w), np.unique(policy))
        policy = pd.get_dummies(policy).values
        for _diff in diff:
            policy = np.insert(policy, _diff, 0, axis=1)
    indicator_matrix = policy * treatment_matrix
    outcome_matrix = np.expand_dims(y, axis=1) * treatment_matrix
    decision_value = np.mean(np.sum(indicator_matrix * (outcome_matrix / ps), axis=1))
    return decision_value


def dr_value(y: np.ndarray, w: np.ndarray, policy: np.ndarray,
             mu: np.ndarray, ps: Optional[np.ndarray]=None) -> float:
    """Decision Value Estimator based on Doubly Robust method.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = (n_samples)
        Treatment assignment indicators.

    policy: array-like of shape = (n_samples)
        Estimated decision model.

    ps: array-like of shape = (n_samples)
        Estimated propensity scores.

    mu: array-like of shape = (n_samples, n_treatments), optional
        Estimated potential outcome for each treatment.

    Returns
    -------
    decision_value: float
        Estimated decision value using Doubly Robust method.

    References
    ----------
    [1] A. Schuler, M. Baiocchi, R. Tibshirani, N. Shah: A comparison of methods for model selection when estimating individual treatment effects, 2018.

    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray.")
    if not isinstance(w, np.ndarray):
        raise TypeError("w must be a numpy.ndarray.")
    if not isinstance(policy, np.ndarray):
        raise TypeError("policy must be a numpy.ndarray.")
    if not isinstance(mu, np.ndarray):
        raise TypeError("mu must be a numpy.ndarray.")
    if ps is None:
        trts_probs = pd.get_dummies(w).mean(axis=0).values
        ps = np.ones((w.shape[0], np.unique(w).shape[0])) * np.expand_dims(trts_probs, axis=0)
    else:
        assert (np.max(ps) < 1) and (np.min(ps) > 0), "ps must be strictly between 0 and 1."

    treatment_matrix = pd.get_dummies(w).values
    policy = pd.get_dummies(policy).values
    diff = np.setdiff1d(np.unique(w), np.unique(policy))
    for _diff in diff:
        policy = np.insert(policy, _diff, 0, axis=1)
    indicator_matrix = policy * treatment_matrix
    outcome_matrix = np.expand_dims(y, axis=1) * treatment_matrix
    decision_value = np.mean(np.sum(treatment_matrix * mu + indicator_matrix * (outcome_matrix - mu) / ps, axis=1))
    return decision_value
