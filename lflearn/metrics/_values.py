"""Evaluation metrics for treatment optimizations."""
from typing import Optional

import numpy as np
import pandas as pd


def expected_response(y: np.ndarray, w: np.ndarray, policy: np.ndarray,
                      mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None) -> float:
    """Estimate expected response.

    Parameters
    ----------
    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).

    w : array-like, shape = [n_samples]
        The treatment assignment. The values should be binary.

    policy: array-like of shape = [n_samples]
        Treatment policy.

    mu: array-like, shape = [n_samples]
        The estimated potential outcomes.

    ps: array-like, shape = [n_samples]
        The estimated propensity scores.


    Returns
    -------
    expected_response: float
        Estimated expected_response.

    References
    ----------
    [1] Yan Zhao, Xiao Fang, and David Simchi-Levi. "Uplift Modeling with Multiple Treatments and
        General Response Types." In Proceedings of the SIAM International Conference on Data Mining, 2017.

    """
    # preprocess potential outcome and propensity estimations.
    mu = np.zeros((w.shape[0], np.unique(w).shape[0])) if mu is None else mu
    ps = pd.get_dummies(w).mean(axis=0).values if ps is None else ps
    # estimate expected response of the policy.
    indicator = np.array(w == policy, dtype=int)
    expected_response = np.mean(mu[np.arange(w.shape[0]), policy]
                                + (y - mu[np.arange(w.shape[0]), policy]) * indicator / ps[w])

    return expected_response
