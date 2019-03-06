"""This module contains metrics to assess performance of uplift models."""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
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

    mu: array-like, shape = [n_samples], optional (default=None)
        The estimated potential outcomes. If None, then MSE is estimated by the IPS method.

    ps: array-like, shape = [n_samples], optional (default=None)
        The estimated propensity scores. If None, then the given data is regarded as RCT data and
        MSE is estimated without dealing with the effect of confounders.

    gamma: float, (default=0.0)
        The switching hyper-parameter.

    Returns
    -------
    mse: float
        The estimated mean squared error for the ITE.

    References
    ----------
    [1] Yuta Saito, Hayato Sakata, and Kazuhide Nakata. "Doubly Robust Prediction
        and Evaluation Methods Improve Uplift Modeling for Observaional Data",
        In Proceedings of the SIAM International Conference on Data Mining, 2019.

    """
    # preprocess potential outcome and propensity estimations.
    mu = np.zeros((w.shape[0], 2)) if mu is None else mu
    ps = pd.get_dummies(w).mean(axis=0).values[w] if ps is None else ps
    # calcurate switch doubly robust outcome.
    direct_estimates = mu[:, 1] - mu[:, 0]
    transformed_outcome = w * (y - mu[:, 1]) / ps[:, 1] - (1 - w) * (y - mu[:, 0]) / ps[:, 0] + direct_estimates
    transformed_outcome[(w == 1) & (ps[:, 1] < gamma)] = direct_estimates[(w == 1) & (ps[:, 1] < gamma)]
    transformed_outcome[(w == 0) & (ps[:, 0] < gamma)] = direct_estimates[(w == 0) & (ps[:, 0] < gamma)]
    # calcurate mse.
    mse = mean_squared_error(transformed_outcome, ite_pred)
    return mse


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

    mu: array-like, shape = [n_samples], optional (default=None)
        The estimated potential outcomes. If None, then expected response is estimated
        by the IPS method.

    ps: array-like, shape = [n_samples], optional (default=None)
        The estimated propensity scores. If None, then the given data is regarded as RCT data and
        expected response is estimated without dealing with the effect of confounders.

    Returns
    -------
    expected_response: float
        Estimated expected_response.

    References
    ----------
    [1] Yan Zhao, Xiao Fang, and David Simchi-Levi. "Uplift Modeling with Multiple Treatments and
        General Response Types." In Proceedings of the SIAM International Conference on Data Mining, 2017.

    """
    num_data, num_trts = w.shape[0], np.unique(w).shape[0]
    # preprocess potential outcome and propensity estimations.
    mu = np.zeros(num_data) if mu is None else mu[np.arange(num_data), w]
    ps = pd.get_dummies(w).mean(axis=0).values[w] if ps is None else ps[np.arange(num_data), w]
    # estimate expected response of the policy.
    indicator = np.array(w == policy, dtype=int)
    expected_response = np.mean(mu + w * (y - mu) * indicator / ps)

    return expected_response


def uplift_frame(ite_pred: np.ndarray, policy: np.ndarray,
                 y: Optional[np.ndarray]=None, w: Optional[np.ndarray]=None,
                 mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
                 gamma: float=0.0, real_world: bool=True) -> DataFrame:
    """Create uplift frame, which is used to plot uplift curves or modified uplift curve.

    Parameters
    ----------
    ite_pred: array-like of shape = [n_samples, n_trts - 1]
        The predicted values of Individual Treatment Effects.

    policy: array-like of shape = [n_samples]
        Treatment policy.

    y : array-like, shape = [n_samples], optional (default=None)
        The target values (class labels in classification, real numbers in
        regression).

    w : array-like, shape = [n_samples], optional (default=None)
        The treatment assignment. The values should be binary.

    mu: array-like, shape = [n_samples], optional (default=None)
        The estimated potential outcomes. If None, then average uplift and expected response
        are estimated by the IPS method.

    ps: array-like, shape = [n_samples], optional (default=None)
        The estimated propensity scores. If None, then the given data is regarded as RCT data and
        average uplift and expected response are estimated without dealing with the effect of confounders.

    gamma: float (default=0.0)
        The switching hyper-parameter.

    real_world: bool (default=True)
        Whether the given data is real-world or synthetic. If True, average uplift and expected response are
        estimated from the ovserved variables. If False, then we have the groud truth of them.


    Returns
    -------
    df: Dataframe
        The uplift frame.
        For real world datasets, columns are ["y", "w", "policy", "ite_pred", "lift", "value"].
        For synthetic datasets, columns are ["mu", "policy", "true_ite", "lift", "value"].

    """
    if real_world:
        df = _create_uplift_frame_from_real_data(ite_pred, policy, y, w, mu, ps, gamma)
    else:
        df = _create_uplift_frame_from_synthetic(ite_pred, policy, mu)

    return df


def _create_uplift_frame_from_real_data(ite_pred: np.ndarray, policy: np.ndarray,
                                        y: np.ndarray, w: np.ndarray,
                                        mu: Optional[np.ndarray], ps: Optional[np.ndarray], gamma: float) -> List:
    """Create uplift frame from real-world data."""
    # initialize variables.
    num_data = w.shape[0]
    num_trts = np.unique(w).shape[0]
    treat_outcome = 0
    baseline_outcome = 0
    treat_value = 0.0
    # preprocess potential outcomes and propensity estimations.
    ps = np.ones((num_data, num_trts)) * pd.get_dummies(w).values.mean(axis=0) if ps is None else ps
    mu = np.zeros((num_data, num_trts)) if mu is None else mu
    # estimate the value of the baseline policy.
    baseline_value = np.sum(mu[:, 0] + np.array(w == 0, dtype=int) * (y - mu[:, 0]) / ps[:, 0])

    # sort data according to the predicted ite values.
    sorted_idx = np.argsort(-ite_pred) if ite_pred.ndim == 1 else np.argsort(-np.max(ite_pred, axis=1))
    y, w, ite_pred, policy, ps, mu = \
        y[sorted_idx], w[sorted_idx], ite_pred[sorted_idx], \
        policy[sorted_idx], ps[sorted_idx], mu[sorted_idx]

    test_data: list = []
    # The best treatment for each data witout the baseline treatment.
    best_trt = np.argmax(ite_pred[:, 1:], axis=1) if num_trts != 2 else np.ones(num_data, dtype=int)
    # Adding all zero value column to have the same number of columns with num_trts.
    ite_pred = np.c_[np.zeros((ite_pred.shape[0],)), ite_pred]
    # estimate lift and value at each treatment point.
    for y_iter, w_iter, ps_iter, mu_iter, pol_iter, trt_iter, ite_iter in zip(y, w, ps, mu, policy, best_trt, ite_pred):
        # update the lift estimates.
        indicator = np.int((w_iter == trt_iter) & (ps_iter[trt_iter] > gamma))
        if w_iter == 0:
            treat_outcome += mu_iter[trt_iter]
            baseline_outcome += mu_iter[0] + np.int(ps_iter[0] > gamma) * (y_iter - mu_iter[0]) / ps_iter[0]
        else:
            treat_outcome += mu_iter[trt_iter] + indicator * (y_iter - mu_iter[trt_iter]) / ps_iter[trt_iter]
            baseline_outcome += mu_iter[0]

        # update the value estimates.
        indicator = np.int((w_iter == pol_iter) & (ps_iter[pol_iter] > gamma))
        if pol_iter == 0:
            baseline_outcome += mu_iter[0] + indicator * (y_iter - mu_iter[0]) / ps_iter[0]
        else:
            baseline_value -= mu_iter[0]
            treat_value += mu_iter[pol_iter] + indicator * (y_iter - mu_iter[pol_iter]) / ps_iter[pol_iter]

        # calc lift and value.
        lift, value = treat_outcome - baseline_outcome, (treat_value + baseline_value) / num_data
        test_data.append([y_iter, w_iter, pol_iter, ite_iter[pol_iter], lift, value])

    # convert to dataframe.
    df = DataFrame(test_data, columns=["y", "w", "policy", "ite_pred", "lift", "value"])
    # calc baseline lift
    df["baseline_lift"] = df.index.values * df.loc[df.shape[0] - 1, "lift"] / df.shape[0]

    return df


def _create_uplift_frame_from_synthetic(ite_pred: np.ndarray, policy: np.ndarray, mu: np.ndarray) -> List:
    """Create uplift frame from synthetic data."""
    # initialize variables.
    num_data = mu.shape[0]
    treat_outcome = 0
    baseline_outcome = 0
    treat_value = 0.0
    # estimate the value of the baseline policy.
    baseline_value = np.sum(mu[:, 0])

    # sort data according to the predicted ite values.
    sorted_idx = np.argsort(-ite_pred) if ite_pred.ndim == 1 else np.argsort(-np.max(ite_pred, axis=1))
    policy, mu, ite_pred = policy[sorted_idx], mu[sorted_idx], ite_pred[sorted_idx]

    test_data: list = []
    # Adding all zero value column to have the same number of columns with num_trts.
    ite_pred = np.c_[np.zeros((ite_pred.shape[0],)), ite_pred]
    # estimate lift and value at each treatment point.
    for mu_iter, pol_iter, ite_iter in zip(mu, policy, ite_pred):
        # update the lift and value estimates.
        treat_value += mu_iter[pol_iter]
        baseline_value -= mu_iter[0]
        treat_outcome += mu_iter[pol_iter]
        baseline_outcome += mu_iter[0]

        # calc lift and value.
        lift, value = treat_outcome - baseline_outcome, (treat_value + baseline_value) / num_data
        test_data.append([mu_iter[pol_iter], pol_iter, ite_iter[pol_iter], lift, value])

    # convert to dataframe.
    df = DataFrame(test_data, columns=["mu", "policy", "true_ite", "lift", "value"])
    # calc baseline lift
    df["baseline_lift"] = df.index.values * df.loc[df.shape[0] - 1, "lift"] / df.shape[0]

    return df


def optimal_uplift_frame(y: np.ndarray, w: np.ndarray,
                         mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
                         ite_true: Optional[np.ndarray]=None, gamma: float=0.) -> DataFrame:
    """Create the uplift frame of the optimal uplift model.

    Parameters
    ----------
    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).

    w : array-like, shape = [n_samples]
        The treatment assignment. The values should be binary.

    mu: array-like, shape = [n_samples], optional (default=None)
        The estimated potential outcomes. If None, then average uplift and expected response
        are estimated by the IPS method.

    ps: array-like, shape = [n_samples], optional (default=None)
        The estimated propensity scores. If None, then the given data is regarded as RCT data and
        average uplift and expected response are estimated without dealing with the effect of confounders.

    ite_true: array-like of shape = [n_samples, n_trts - 1]
        The true values of the ITE. If None, then the given data is regarded as real-world and
        average uplift and expected response are estimated from the ovserved variables.

    gamma: float (default=0.0)
        The switching hyper-parameter.

    Returns
    -------
    df: Dataframe
        The uplift frame of the optimal policy.
        For real world datasets, columns are ["y", "w", "policy", "ite_pred", "lift", "value"].
        For synthetic datasets, columns are ["mu", "policy", "true_ite", "lift", "value"].

    """
    if ite_true is None:
        num_data, num_trts = w.shape[0], np.unique(w).shape[0]
        # preprocess potential outcome and propensity estimations.
        ps = np.ones((num_data, num_trts)) * pd.get_dummies(w).values.mean(axis=0) if ps is None else ps
        mu = np.zeros((num_data, num_trts)) if mu is None else mu
        # estimate potential outcomes of the given test data.
        _y, _w = np.expand_dims(y, axis=1), pd.get_dummies(w).values
        value = mu + np.array(ps > gamma, dtype=int) * _w * (_y - mu) / ps
        # calc the oracle ite values and the optimal policy.
        optimal_ite, optimal_policy = value[:, 1] - value[:, 0], np.argmax(value, axis=1)
        # create the optimal uplift frame for the given real-world dataset.
        optimal_frame = _create_uplift_frame_from_real_data(optimal_ite, optimal_policy, y, w, mu, ps, gamma)

    else:
        # Adding all zero value column to have the same number of columns with num_trts.
        ite_true = np.c_[np.zeros((ite_true.shape[0],)), ite_true]
        # calc the optimal policy.
        optimal_policy = np.argmax(ite_true, axis=1)
        # create the optimal uplift frame for the given synthetic dataset.
        optimal_frame = _create_uplift_frame_from_synthetic(ite_true, optimal_policy, mu)

    return optimal_frame
