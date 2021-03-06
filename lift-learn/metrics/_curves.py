"""Metrics to assess performance on ite prediction task."""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

from plotly.graph_objs import Bar, Box, Figure, Layout, Scatter
from plotly.offline import init_notebook_mode, iplot, plot


def uplift_frame(ite_pred: np.ndarray,
                 y: Optional[np.ndarray]=None, w: Optional[np.ndarray]=None,
                 mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
                 gamma: float=0.0, real_world: bool=True) -> DataFrame:
    """Create uplift frame, which is used to plot uplift curves or modified uplift curve.

    Parameters
    ----------
    ite_pred: array-like of shape = (n_samples, n_trts - 1)
        Estimated Individual Treatment Effects.

    policy: array-like of shape = (n_samples,)
        Estimated Individual Treatment Effects.

    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = (n_samples)
        Treatment assignment indicators.

    mu: array-like of shape = (n_samples, n_trts), optional
        Estimated potential outcomes for the treated.

    ps: array-like of shape = (n_samples), optional
        Estimated propensity scores.

    threshold: float (default=0.0), optional
        A cipping threshold for estimated propensity scores.

    gamma: float (default=0.0), optional
        A switching parameter for doubly robust method.

    real_world: bool (default=True), optional
        If the given data is real-world or synthetic.

    Returns
    -------
    df: Dataframe of shape = (n_samples, 9)
        An uplift frame.

    References
    ----------
    [1] 有賀康顕, 中山心太, 西林孝: 仕事で始める機械学習, オライリー・ジャパン, 2017.

    """
    if ite_pred.ndim == 1:
        ite_pred = np.c_[np.zeros(ite_pred.shape[0]), ite_pred]
        policy = np.ones(ite_pred.shape[0], dtype=int)
    else:
        policy = np.argmax(ite_pred[:, 1], axis=1) + 1

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
    """Create uplift from from synthetic data."""
    # Initialize variables.
    num_data = w.shape[0]
    num_trts = np.unique(w).shape[0]
    num_treated_data = 0
    num_baseline_data = 0
    treat_outcome = 0
    baseline_outcome = 0
    treat_value = 0.0
    ps = np.ones((num_data, num_trts)) * pd.get_dummies(w).values.mean(axis=0) if ps is None else ps
    mu = np.zeros((num_data, num_trts)) if mu is None else mu
    baseline_indicator = np.array(w == 0, dtype=int)
    baseline_value = np.sum(mu[:, 0] + baseline_indicator * (y - mu[:, 0]) / ps[:, 0])

    # Sort data according to the ite from the baseline treatment.
    sorted_ite_from_baseline = np.argsort(np.max(ite_pred[:, 1:], axis=1))
    y, w, policy, ps, mu, ite_pred = \
        y[sorted_ite_from_baseline][::-1], \
        w[sorted_ite_from_baseline][::-1], \
        policy[sorted_ite_from_baseline][::-1], \
        ps[sorted_ite_from_baseline][::-1], \
        mu[sorted_ite_from_baseline][::-1], \
        ite_pred[sorted_ite_from_baseline][::-1]

    # Estimate lift and value at each treatment point.
    test_data: list = []
    for y_iter, w_iter, ps_iter, mu_iter, pol_iter, ite_iter in zip(y, w, ps, mu, policy, ite_pred):
        if w_iter == 0:
            num_baseline_data += 1
            if ps_iter[0] < gamma:
                baseline_value -= mu_iter[0]
                baseline_outcome += mu_iter[0]
            else:
                baseline_value -= mu_iter[0] + (y_iter - mu_iter[0]) / ps_iter[0]
                baseline_outcome += mu_iter[0] + (y_iter - mu_iter[0]) / ps_iter[0]
            treat_value += mu_iter[pol_iter]
            treat_outcome += mu_iter[pol_iter]

        else:
            num_treated_data += 1
            indicator = np.int(w_iter == pol_iter)
            if ps_iter[pol_iter] < gamma:
                treat_value += mu_iter[pol_iter]
                treat_outcome += mu_iter[pol_iter]
            else:
                treat_value += mu_iter[pol_iter] + indicator * (y_iter - mu_iter[pol_iter]) / ps_iter[pol_iter]
                treat_outcome += mu_iter[pol_iter] + indicator * (y_iter - mu_iter[pol_iter]) / ps_iter[pol_iter]
            baseline_value -= mu_iter[0]
            baseline_outcome += mu_iter[0]

        # calc lift at each intervention point.
        lift = treat_outcome - baseline_outcome
        # calc value at each intervention point.
        value = (treat_value + baseline_value) / num_data

        test_data.append([y_iter, w_iter, pol_iter, ite_iter[pol_iter], lift, value])

    return test_data


def _create_uplift_frame_from_synthetic(ite_pred: np.ndarray, policy: np.ndarray, mu: np.ndarray) -> List:
    """Create uplift from from synthetic data."""
    # Initialize variables.
    num_data = mu.shape[0]
    treat_outcome = 0
    baseline_outcome = 0
    treat_value = 0.0
    baseline_value = np.sum(mu[:, 0])

    # Sort data according to the ite from the baseline treatment.
    sorted_ite_from_baseline = np.argsort(np.max(ite_pred[:, 1:], axis=1))
    policy, mu, ite_pred = policy[sorted_ite_from_baseline][::-1], \
        mu[sorted_ite_from_baseline][::-1], \
        ite_pred[sorted_ite_from_baseline][::-1]

    # Estimate lift and value at each treatment point.
    test_data: list = []
    n_iter = 0
    for mu_iter, pol_iter, ite_iter in zip(mu, policy, ite_pred):
        n_iter += 1
        treat_value += mu_iter[pol_iter]
        baseline_value -= mu_iter[0]
        treat_outcome += mu_iter[pol_iter]
        baseline_outcome += mu_iter[0]

        # calc lift at each intervention point.
        lift = treat_outcome - baseline_outcome
        # calc value at each intervention point.
        value = (treat_value + baseline_value) / num_data

        test_data.append([mu_iter[pol_iter], pol_iter, ite_iter[pol_iter], lift, value])

    return test_data


def optimal_uplift_frame(y: np.ndarray, w: np.ndarray,
                         mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
                         ite: Optional[np.ndarray]=None) -> DataFrame:
    """Create uplift frame, which is used to plot uplift curves or modified uplift curve.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = (n_samples)
        Treatment assignment indicators.

    Returns
    -------
    df: Dataframe of shape = (n_samples, 9)
        The uplift frame of the optimal policy.

    """
    if ite is None:
        # calc value_matrix for the given test data.
        num_data, num_trts = w.shape[0], np.unique(w).shape[0]
        if ps is None:
            ps_matrix = np.ones((num_data, num_trts)) * pd.get_dummies(w).values.mean(axis=0)
        if mu is None:
            mu_matrix = np.zeros((num_data, num_trts))
        outcome_matrix = np.expand_dims(y, axis=1) * w
        value_matrix = mu_matrix + (outcome_matrix - mu_matrix) / ps_matrix

        # calc the optimal ite estimator and the optimal policy.
        optimal_ite = value_matrix - value_matrix[:, 0]
        optimal_policy = np.argmax(value_matrix, axis=1)

        # create the optimal uplift frame.
        optimal_frame = _create_uplift_frame_from_real_data(optimal_ite, optimal_policy, y, w, mu, ps)

    else:
        optimal_policy = np.argmax(ite, axis=1)
        optimal_frame = _create_uplift_frame_from_synthetic(ite, optimal_policy, mu)

    return optimal_frame


def uplift_bar(y: np.ndarray, w: np.ndarray, ite_pred: np.ndarray,
               ps: Union[np.ndarray, None]=None, threshold: float=0.0, design: str="randomized") -> Figure:
    """Plot Uplift Bar.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Observed target values.

    w: array-like of shape = (n_samples)
        Treatment assignment indicators.

    ite_pred: array-like of shape = (n_samples)
        Estimated Individual Treatment Effects.

    ps: array-like of shape = (n_samples)
        Estimated propensity scores.

    threshold: float (default=0.0)
        A clipping threshold for the estimated propensity score.

    design: string in ["randomized", "observational"] (default="randomized")
        Whether the data gatherd through experimental or observational study.

    Returns
    -------
    fig: Figure
        An uplift bar.

    References
    ----------
    [1] 有賀康顕, 中山心太, 西林孝: 仕事で始める機械学習, オライリー・ジャパン, 2017.

    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray.")
    if not isinstance(w, np.ndarray):
        raise TypeError("w must be a numpy.ndarray.")
    if not isinstance(ite_pred, np.ndarray):
        raise TypeError("ite_pred must be a numpy.ndarray.")
    if design not in ["randomized", "observational"]:
        raise ValueError("Invalid value for design. Allowing string values are 'randomized', 'observational'.")

    # sort data according to the estimated ITEs.
    if design == "observational":
        if not isinstance(ps, np.ndarray):
            raise TypeError("ps must be a numpy.ndarray.")
        if not isinstance(threshold, float):
            raise TypeError("threshold must be a float.")
        assert (np.max(ps) < 1) and (np.min(ps) > 0), "ps must be strictly between 0 and 1."
        assert (threshold <= 1) or (threshold >= 0), "threshold must be strictly between 0 and 1."

        df = DataFrame(np.vstack((y, w, ite_pred, ps)).T,
                       columns=["y", "w", "ite_pred", "ps"]).sort_values(by="ite_pred", ascending=False).reset_index(drop=True)
        qdf = DataFrame(columns=("y_treat", "y_control"))
    else:
        df = DataFrame(np.vstack((y, w, ite_pred)).T,
                       columns=["y", "w", "ite_pred"]).sort_values(by="ite_pred", ascending=False).reset_index(drop=True)
        qdf = DataFrame(columns=("y_treat", "y_control"))

    # divide data into deciles.
    for n in np.arange(10):
        start = (n * df.shape[0] / 10)
        end = ((n + 1) * df.shape[0] / 10) - 1
        _df = df.loc[start:end]

        if design == "observational":
            ps_treat = np.clip(_df[_df.w == 1].ps.values, threshold, 1.0)
            ps_control = 1.0 - np.clip(_df[_df.w == 0].ps.values, 0., 1.0 - threshold)
            y_treat = _df[_df.w == 1].y.values
            y_treat_estimator = np.sum(y_treat / ps_treat) / np.sum(1.0 / ps_treat)
            y_control = _df[_df.w == 0].y.values
            y_control_estimator = np.sum(y_control / ps_control) / np.sum(1.0 / ps_control)

        elif design == "randomized":
            y_treat_estimator = np.mean(_df[_df.w == 1].y.values)
            y_control_estimator = np.mean(_df[_df.w == 0].y.values)

        label = f"{n * 10}%~{(n + 1) * 10}%"
        qdf.loc[label] = [y_treat_estimator, y_control_estimator]

    trace1 = Bar(x=qdf.index.tolist(),
                 y=qdf.y_treat.tolist(), name="Treatment Group")
    trace2 = Bar(x=qdf.index.tolist(),
                 y=qdf.y_control.tolist(), name="Control Group")

    layout = Layout(barmode="group", yaxis={"title": "Estimated Strata Outcome"}, xaxis={"title": "Uplift Score Percentile"})
    fig = Figure(data=[trace1, trace2], layout=layout)
    return fig


def treat_percentile_bar(w: np.ndarray, ite_pred: np.ndarray) -> Figure:
    """Plot treat percentile bar.

    Parameters
    ----------
    w: array-like of shape = (n_samples)
        Treatment assignment indicators.

    ite_pred: array-like of shape = (n_samples)
        Estimated Individual Treatment Effects.

    Returns
    -------
    fig: Figure
        An treat percentile bar.

    """
    if not isinstance(w, np.ndarray):
        raise TypeError("w must be a numpy.ndarray.")
    if not isinstance(ite_pred, np.ndarray):
        raise TypeError("ite_pred must be a numpy.ndarray.")

    # sort data according to the estimated ITEs.
    df = DataFrame(np.vstack((w, ite_pred)).T,
                   columns=["w", "ite_pred"]).sort_values(by="ite_pred", ascending=False).reset_index(drop=True)
    qdf = DataFrame(columns=["pct_treated"])

    # divide data into deciles.
    for n in np.arange(10):
        start = (n * df.shape[0] / 10)
        end = ((n + 1) * df.shape[0] / 10) - 1
        _df = df.loc[start:end]

        pct_treated = _df.treat.mean()

        label = f"{n * 10}%~{(n + 1) * 10}%"
        qdf.loc[label] = [pct_treated]

    trace = Bar(x=qdf.index.tolist(), y=qdf.pct_treated.tolist())
    layout = Layout(yaxis={"title": "Propensity of being treated"}, xaxis={"title": "Uplift Score Percentile"})
    fig = Figure(data=[trace], layout=layout)
    return fig


def uplift_curve(uplift_frame: DataFrame, name: str, rounds: int=3) -> Figure:
    """Plot Uplift Curve.

    Parameters
    ----------
    uplift_frame: DataFrame of shape = (n_samples, 9)
        An uplift frame.

    Returns
    -------
    fig: Figure
        An uplift curve.

    References
    ----------
    [1] 有賀康顕, 中山心太, 西林孝: 仕事で始める機械学習, オライリー・ジャパン, 2017.
    [2] P. Gutierrez, J. Y. Gerardy: Causal Inference and Uplift Modeling A review of the literature, 2016.

    """
    if not isinstance(uplift_frame, DataFrame):
        raise TypeError("uplift_frame must be a pandas.DataFrame.")

    # calc Area Under Uplift Curve
    auuc = np.round(np.mean(uplift_frame.lift.values - uplift_frame.baseline_lift.values), rounds)

    # Lift Curve sorted by Uplift Rank
    curve1 = Scatter(x=np.arange(uplift_frame.shape[0]) / uplift_frame.shape[0],
                     y=np.ravel(uplift_frame.lift.values), name=name, line=dict(width=4))
    curve2 = Scatter(x=np.arange(uplift_frame.shape[0]) / uplift_frame.shape[0],
                     y=np.ravel(uplift_frame.baseline_lift.values), name="Baseline")
    layout = Layout(title=f"Uplift Curve\n(AUUC = {auuc})", yaxis=dict(title="Average Uplift", titlefont=dict(size=15)),
                    xaxis=dict(title="Proportion of Treated Individuals", titlefont=dict(size=15), tickfont=dict(size=15)))
    fig = Figure(data=[curve1, curve2], layout=layout)
    return fig


def modified_uplift_curve(uplift_frame: DataFrame, name: str, rounds: int=3) -> Figure:
    """Plot Modified Uplift Curve.

    Parameters
    ----------
    uplift_frame: DataFrame of shape = (n_samples, 9)
        An uplift frame.

    Returns
    -------
    fig: Figure
        An modified uplift curve.

    References
    ----------
    [1] Y. Zhao, X. Fang, D. S. Levi: Uplift modeling with multiple treatments and general response types, 2017.

    """
    if not isinstance(uplift_frame, DataFrame):
        raise TypeError("uplift_frame must be a pandas.DataFrame.")

    # calc average value.
    aumuc = np.round(np.mean(uplift_frame.value.values), rounds)

    # Lift Curve sorted by Uplift Score
    curve1 = Scatter(x=np.arange(uplift_frame.shape[0]) / uplift_frame.shape[0],
                     y=np.ravel(uplift_frame.value.values), name=name, line=dict(width=4))
    layout = Layout(title=f"Modified Uplift Curve\n(AUMUC = {aumuc})",
                    yaxis=dict(title="Expected Response", titlefont=dict(size=15)),
                    xaxis=dict(title="Proportion of Treated Individuals",
                               titlefont=dict(size=15), tickfont=dict(size=15)))
    fig = Figure(data=[curve1], layout=layout)
    return fig


def compare_uplift_curves(uplift_frames: List[DataFrame], name_list: List[str]=[]) -> Figure:
    """Compare several uplift curves.

    Parameters
    ----------
    uplift_frames: list of DataFrames of shape = (n_samples, 9)
        An uplift frame.

    name_list: list
        Names of modeling methods.

    Returns
    -------
    fig: Figure
        Uplift curves.

    """
    if not isinstance(uplift_frames, list):
        raise TypeError("uplift_frames must be a list of pandas.DataFrames.")
    if not isinstance(name_list, list):
        raise TypeError("name_list must be a list of strings.")

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
        Modified uplift curves.

    """
    if not isinstance(uplift_frames, list):
        raise TypeError("uplift_frames must be a list of pandas.DataFrames.")
    if not isinstance(name_list, list):
        raise TypeError("name_list must be a list of strings.")

    curve_list = [Scatter(x=np.arange(df.shape[0]) / df.shape[0],
                          y=np.ravel(df.value.values), name=name, line=dict(width=4))
                  for df, name in zip(uplift_frames, name_list)]
    layout = Layout(title="Modified Uplift Curve",
                    yaxis=dict(title="Expected Response", titlefont=dict(size=15)),
                    xaxis=dict(title="Proportion of Treated Individuals",
                               titlefont=dict(size=15), tickfont=dict(size=15)))
    fig = Figure(data=curve_list, layout=layout)
    return fig
