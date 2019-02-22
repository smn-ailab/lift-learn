"""Metrics to assess performance on ite prediction task."""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

from plotly.graph_objs import Bar, Box, Figure, Layout, Scatter


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

    mu: array-like, shape = [n_samples]
        The estimated potential outcomes.

    ps: array-like, shape = [n_samples]
        The estimated propensity scores.

    gamma: float, optional (default=0.0)
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

    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).

    w : array-like, shape = [n_samples]
        The treatment assignment. The values should be binary.

    mu: array-like, shape = [n_samples]
        The estimated potential outcomes.

    ps: array-like, shape = [n_samples]
        The estimated propensity scores.

    gamma: float (default=0.0), optional
        The switching hyper-parameter.

    real_world: bool (default=True), optional
        Whether the given data is real-world or synthetic.

    Returns
    -------
    df: Dataframe of shape = [n_samples, ]
        The uplift frame.

    """
    if real_world:
        test_data = _create_uplift_frame_from_real_data(ite_pred, policy, y, w, mu, ps, gamma)
    else:
        test_data = _create_uplift_frame_from_synthetic(ite_pred, policy, mu)

    # convert to dataframe.
    df = DataFrame(test_data)
    df.columns = [["y", "w", "policy", "ite_pred", "lift", "value"]] if real_world else [["mu", "policy", "true_ite", "lift", "value"]]

    # calc baseline lift
    df["baseline_lift"] = df.index.values * df.loc[df.shape[0] - 1, "lift"][0] / df.shape[0]

    return df


def _create_uplift_frame_from_real_data(ite_pred: np.ndarray, policy: np.ndarray,
                                        y: np.ndarray, w: np.ndarray,
                                        mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None, gamma: float=0.0,) -> List:
    """Create uplift frame from synthetic data."""
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
    sorted_idx = np.argsort(ite_pred) if ite_pred.ndim == 1 else np.argsort(np.max(ite_pred, axis=1))
    y, w, ite_pred, policy, ps, mu = \
        y[sorted_idx][::-1], w[sorted_idx][::-1], \
        ite_pred[sorted_idx][::-1], policy[sorted_idx][::-1], \
        ps[sorted_idx][::-1], mu[sorted_idx][::-1]

    test_data: list = []
    # The best treatment for each data witout the baseline treatment.
    best_trt = np.argmax(ite_pred[:, 1:], axis=1) if num_trts != 2 else np.ones(num_data, dtype=int)
    # Adding all zero value column.
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

    return test_data


def _create_uplift_frame_from_synthetic(ite_pred: np.ndarray, policy: np.ndarray, mu: np.ndarray) -> List:
    """Create uplift frame from from synthetic data."""
    # initialize variables.
    num_data = mu.shape[0]
    treat_outcome = 0
    baseline_outcome = 0
    treat_value = 0.0
    # estimate the value of the baseline policy.
    baseline_value = np.sum(mu[:, 0])

    # sort data according to the predicted ite values.
    sorted_idx = np.argsort(ite_pred) if ite_pred.ndim == 1 else np.argsort(np.max(ite_pred, axis=1))
    policy, mu, ite_pred = policy[sorted_idx][::-1], mu[sorted_idx][::-1], ite_pred[sorted_idx][::-1]

    test_data: list = []
    # Adding all zero value column.
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

    return test_data


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

    mu: array-like, shape = [n_samples]
        The estimated potential outcomes.

    ps: array-like, shape = [n_samples]
        The estimated propensity scores.

    ite_true: array-like of shape = [n_samples, n_trts - 1]
        The true values of the ITE.

    gamma: float (default=0.0), optional
        The switching hyper-parameter.

    Returns
    -------
    df: Dataframe of shape = [n_samples, ]
        The uplift frame of the optimal policy.

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
        optimal_ite = value[:, 1] - value[:, 0]
        optimal_policy = np.argmax(value, axis=1)

        # create the optimal uplift frame for the given real-world dataset.
        optimal_frame = _create_uplift_frame_from_real_data(optimal_ite, optimal_policy, y, w, mu, ps)

    else:
        # Adding all zero value column.
        ite_true = np.c_[np.zeros((ite_true.shape[0],)), ite_true]
        # calc the optimal policy.
        optimal_policy = np.argmax(ite_true, axis=1)

        # create the optimal uplift frame for the given synthetic dataset.
        optimal_frame = _create_uplift_frame_from_synthetic(ite_true, optimal_policy, mu)

    # convert to dataframe.
    df = DataFrame(optimal_frame)
    df.columns = [["y", "w", "policy", "ite_pred", "lift", "value"]] if ite_true is None else [["mu", "policy", "true_ite", "lift", "value"]]

    # calc baseline lift
    df["baseline_lift"] = df.index.values * df.loc[df.shape[0] - 1, "lift"][0] / df.shape[0]

    return df


def uplift_bar(ite_pred: np.ndarray, policy: np.ndarray, y: np.ndarray, w: np.ndarray,
               mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
               gamma: float=0.0, real_world: bool=True) -> Figure:
    """Plot Uplift Bar.

    Parameters
    ----------
    ite_pred: array-like of shape = [n_samples, n_trts - 1]
        The predicted values of Individual Treatment Effects.

    policy: array-like of shape = [n_samples]
        Treatment policy.

    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).

    w : array-like, shape = [n_samples]
        The treatment assignment. The values should be binary.

    mu: array-like, shape = [n_samples]
        The estimated potential outcomes.

    ps: array-like, shape = [n_samples]
        The estimated propensity scores.

    gamma: float (default=0.0), optional
        The switching hyper-parameter.

    real_world: bool (default=True), optional
        Whether the given data is real-world or synthetic.

    Returns
    -------
    fig: Figure
        The uplift bar.

    """
    num_data, num_trts = w.shape[0], np.unique(w).shape[0]
    # preprocess potential outcome and propensity estimations.
    ps = np.ones((num_data, num_trts)) * pd.get_dummies(w).values.mean(axis=0) if ps is None else ps
    mu = np.zeros((num_data, num_trts)) if mu is None else mu
    # sort data according to the predicted ite values.
    sorted_idx = np.argsort(ite_pred) if num_trts == 2 else np.argsort(np.max(ite_pred, axis=1))
    if real_world:
        y, w = np.expand_dims(y, axis=1), pd.get_dummies(w).values
        value = mu + np.array(ps > gamma, dtype=int) * w * (y - mu) / ps
    else:
        value = mu
    value, ite_pred = value[sorted_idx][::-1], ite_pred[sorted_idx][::-1]

    # divide data into deciles.
    labels, treat_outcome_list, baseline_outcome_list = [], [], []
    best_trt = np.argmax(ite_pred[:, 1:], axis=1) if num_trts != 2 else np.ones(num_data, dtype=int)

    # estimate the ATE of each stratum.
    for n in np.arange(10):
        start, end = np.int(n * num_data / 10), np.int((n + 1) * num_data / 10) - 1
        treat_outcome, baseline_outcome = np.mean(value[start:end, best_trt[start:end]]), np.mean(value[start:end, 0])
        treat_outcome_list.append(treat_outcome)
        baseline_outcome_list.append(baseline_outcome)
        labels.append(f"{n * 10}% ~ {(n + 1) * 10}%")

    trace1 = Bar(x=labels, y=treat_outcome_list, name="Treatment Group")
    trace2 = Bar(x=labels, y=baseline_outcome_list, name="Control Group")

    layout = Layout(barmode="group", yaxis={"title": "Estimated Outcome"}, xaxis={"title": "Top ITE Percentile"})
    fig = Figure(data=[trace1, trace2], layout=layout)
    return fig


def compare_uplift_curves(uplift_frames: List[DataFrame], name_list: List[str]=[]) -> Figure:
    """Compare several uplift curves.

    Parameters
    ----------
    uplift_frames: list of DataFrames of shape = (n_samples, 9)
        The list of uplift frames.

    name_list: list
        The list of names of modeling methods.

    Returns
    -------
    fig: Figure
        The uplift curves.

    """
    curve_list = [Scatter(x=np.arange(df.shape[0]) / df.shape[0],
                          y=np.ravel(df.lift.values), name=name, line=dict(width=4))
                  for df, name in zip(uplift_frames, name_list)]
    curve_list.append(Scatter(x=np.arange(uplift_frames[0].shape[0]) / uplift_frames[0].shape[0],
                              y=np.ravel(uplift_frames[0].baseline_lift.values), name="Baseline"))
    layout = Layout(title=f"Uplift Curve",
                    yaxis=dict(title="Average Uplift", titlefont=dict(size=15)),
                    xaxis=dict(title="Proportion of Treated Individuals",
                               titlefont=dict(size=15), tickfont=dict(size=15)))
    fig = Figure(data=curve_list, layout=layout)
    return fig


def compare_modified_uplift_curves(uplift_frames: List[DataFrame], name_list: List[str]=[]) -> Figure:
    """Compare several modified uplift curves.

    Parameters
    ----------
    uplift_frames: list of DataFrames of shape = (n_samples, 9)
        An uplift frame.

    name_list: list
        Names of modeling methods.

    Returns
    -------
    fig: Figure
        The modified uplift curves.

    """
    curve_list = [Scatter(x=np.arange(df.shape[0]) / df.shape[0],
                          y=np.ravel(df.value.values), name=name, line=dict(width=4))
                  for df, name in zip(uplift_frames, name_list)]
    layout = Layout(title="Modified Uplift Curve",
                    yaxis=dict(title="Expected Response", titlefont=dict(size=15)),
                    xaxis=dict(title="Proportion of Treated Individuals",
                               titlefont=dict(size=15), tickfont=dict(size=15)))
    fig = Figure(data=curve_list, layout=layout)
    return fig
