"""This module contains visualization tools to assess performance of uplift models."""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from plotly.graph_objs import Bar, Box, Figure, Layout, Scatter


def uplift_bar(ite_pred: np.ndarray, policy: np.ndarray, y: np.ndarray, w: np.ndarray,
               mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
               gamma: float=0.0, num_strata: int=10, real_world: bool=True) -> Figure:
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

    mu: array-like, shape = [n_samples], optional (default=None)
        The estimated potential outcomes. If None, then ATE is estimated by the IPS method.

    ps: array-like, shape = [n_samples], optional (default=None)
        The estimated propensity scores. If None, then the given data is regarded as RCT data and
        ATE is estimated without dealing with the effect of confounders.

    gamma: float (default=0.0)
        The switching hyper-parameter.

    num_strata: int (default=10)
        The number of strata in stratified ATE estimation.

    real_world: bool (default=True)
        Whether the given data is real-world or synthetic. If True, ATE is estimated from
        the ovserved variables. If False, then we have the groud truth of it.

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
    for n in np.arange(num_strata):
        start, end = np.int(n * num_data / num_strata), np.int((n + 1) * num_data / num_strata) - 1
        treat_outcome, baseline_outcome = np.mean(value[start:end, best_trt[start:end]]), np.mean(value[start:end, 0])
        treat_outcome_list.append(treat_outcome)
        baseline_outcome_list.append(baseline_outcome)
        labels.append(f"{np.int(n * 100 / num_strata)}% ~ {np.int((n + 1) * 100 / num_strata)}%")

    trace1 = Bar(x=labels, y=treat_outcome_list, name="Treatment Group")
    trace2 = Bar(x=labels, y=baseline_outcome_list, name="Control Group")

    layout = Layout(barmode="group", yaxis={"title": "Estimated Outcome"}, xaxis={"title": "Top ITE Percentile"})
    fig = Figure(data=[trace1, trace2], layout=layout)
    return fig


def compare_uplift_curves(uplift_frames: List[DataFrame], name_list: List[str]=[]) -> Tuple[Figure, Figure]:
    """Compare several uplift curves and modified uplift curves.

    Parameters
    ----------
    uplift_frames: list of DataFrames of shape = (n_samples, 9)
        The list of uplift frames.

    name_list: list
        The list of names of modeling methods.

    Returns
    -------
    curve_fig, modi_curve_fig: Tuple[Figure, Figure]
        The Figure of uplift curves and the modified uplift curves.

    References
    ----------
    [1] Yan Zhao, Xiao Fang, and David Simchi-Levi. "Uplift Modeling with Multiple Treatments and
        General Response Types." In Proceedings of the SIAM International Conference on Data Mining, 2017.

    """
    # figure of modified uplift curves.
    curve_list = [Scatter(x=np.arange(df.shape[0]) / df.shape[0],
                          y=np.ravel(df.lift.values), name=name, line=dict(width=4))
                  for df, name in zip(uplift_frames, name_list)]
    curve_list.append(Scatter(x=np.arange(uplift_frames[0].shape[0]) / uplift_frames[0].shape[0],
                              y=np.ravel(uplift_frames[0].baseline_lift.values), name="Baseline"))
    curve_layout = Layout(title=f"Uplift Curve",
                          yaxis=dict(title="Average Uplift", titlefont=dict(size=15)),
                          xaxis=dict(title="Proportion of Treated Individuals",
                                     titlefont=dict(size=15), tickfont=dict(size=15)))
    curve_fig = Figure(data=curve_list, layout=curve_layout)

    # figure of modified uplift curves.
    modi_curve_list = [Scatter(x=np.arange(df.shape[0]) / df.shape[0],
                               y=np.ravel(df.value.values), name=name, line=dict(width=4))
                       for df, name in zip(uplift_frames, name_list)]
    modi_curve_layout = Layout(title="Modified Uplift Curve",
                               yaxis=dict(title="Expected Response", titlefont=dict(size=15)),
                               xaxis=dict(title="Proportion of Treated Individuals",
                                          titlefont=dict(size=15), tickfont=dict(size=15)))
    modi_curve_fig = Figure(data=modi_curve_list, layout=modi_curve_layout)

    return curve_fig, modi_curve_fig
