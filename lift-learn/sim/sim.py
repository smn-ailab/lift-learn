"""This Module contains a class to run simulations of uplift modeling on real-world datasets."""
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from causalml.metrics import (compare_modified_uplift_curves,
                              compare_uplift_curves, optimal_uplift_frame,
                              uplift_frame)
from plotly.graph_objs import Box, Figure, Layout


class UpliftSimulator():
    """A class to assess performance of meta learning methods and evaluation metrics for offline uplift modeling."""

    def __init__(self, um_list: List[BaseEstimator], name_list: Optional[List[str]]=None,
                 real_world: bool=True, rct: bool=True, method: str="naive", regression: bool=True) -> None:
        """Initialize Class.

        Parameters
        ----------
        um_list: list of objects
            A list of ITE prediciton models which are to be evaluated in a simulation.

        name_list: list of string, optional
            A list of names of ITE prediciton models.

        """
        self.num = 1
        self.um_list = um_list
        if name_list is None:
            self.name_list = [um.name for um in self.um_list]
        else:
            self.name_list = name_list
        self.real_world, self.rct, self.method, self.regression = real_world, rct, method, regression
        if not self.real_world or self.rct:
            self.name_list = ["Oracle"] + self.name_list
        self.auuc_df, self.aumuc_df, self.ite_df, self.value_df = \
            DataFrame(), DataFrame(), DataFrame(), DataFrame()
        if not self.real_world or self.rct:
            self.lift_df_list, self.value_df_list = \
                [DataFrame() for i in np.arange(len(um_list) + 1)], [DataFrame() for i in np.arange(len(um_list) + 1)]
        else:
            self.lift_df_list, self.value_df_list = \
                [DataFrame() for i in np.arange(len(um_list))], [DataFrame() for i in np.arange(len(um_list))]

    def estimate(self, X: np.ndarray, y_obs: np.ndarray, w: np.ndarray,
                 ps_estimator: ClassifierMixin=LR(C=100, solver="lbfgs", random_state=0),
                 pom_treat: Optional[BaseEstimator]=None, pom_control: Optional[BaseEstimator]=None,
                 mu_treat: Optional[np.ndarray]=None, mu_control: Optional[np.ndarray]=None) -> None:
        """Estimate the potential outcomes and the propensity score.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Test input samples.

        y_obs : array-like of shape = (n_samples)
            Observed target values.(class labels in classification, real numbers in regression).

        w : array-like of shape = (n_samples)
            Treatment assignment indicators.

        pom_treat: object
            The Potential Outcome Model for the treated from which the uplift curve is estimated.

        pom_control: object
            The Potential Outcome Model for the controlled from which the uplift curve is estimated.

        ps_estimator: object (default=LogisticRegression(C=100, solver="lbfgs", random_state=0))
            The Propensity Score estimator.

        """
        self.X, self.y, self.w, self.mu_treat, self.mu_control = \
            X, y_obs, w, mu_treat, mu_control

        if self.real_world:
            if not self.rct:
                # ps estimation for train and test data separately.
                ps_estimator_ = ps_estimator
                ps_estimator_.fit(X, w)
                self.ps = ps_estimator_.predict_proba(X)[:, 1]

            if not self.method == "naive":
                # outcomes models.
                pom_treat_ = pom_treat
                pom_control_ = pom_control
                # treat-control split.
                X_treat, X_control, y_treat, y_control = \
                    self.X[self.w == 1], self.X[self.w == 0], self.y[self.w == 1], self.y[self.w == 0]
                # potential outcomes estimation for test data.
                pom_treat_.fit(X_treat, y_treat)
                pom_control_.fit(X_control, y_control)
                if self.regression:
                    self.mu_treat, self.mu_control = \
                        pom_treat_.predict(X), pom_control_.predict(X)
                else:
                    self.mu_treat, self.mu_control = \
                        pom_treat_.predict_proba(X)[:, 1], pom_control_.predict_proba(X)[:, 1]

        else:
            self.ite = self.mu_treat - self.mu_control

    def split_data(self, test_size: float=0.3, random_state: int=0) -> None:
        """Train/Test Split.

        Parameters
        ----------
        test_size: float, (default=0.3)
            Represent the proportion of the dataset to include in the test split and should be between 0.0 and 1.0.

        random_state: int, (default=0)
            The seed used by the train-test spliter.

        """
        # train-test split.
        if self.real_world:
            if not self.rct:
                self.X_train, self.X_test, self.w_train, self.w_test, self.y_train, self.y_test, \
                    _, self.ps_test, _, self.mu_treat_test, _, self.mu_control_test = \
                    train_test_split(self.X, self.w, self.y, self.ps, self.mu_treat, self.mu_control,
                                     test_size=test_size, random_state=random_state)

            else:
                if not self.method == "naive":
                    self.X_train, self.X_test, self.w_train, self.w_test, self.y_train, self.y_test, \
                        _, self.mu_treat_test, _, self.mu_control_test = \
                        train_test_split(self.X, self.w, self.y, self.mu_treat, self.mu_control,
                                         test_size=test_size, random_state=random_state)

                else:
                    self.X_train, self.X_test, self.w_train, self.w_test, self.y_train, self.y_test = \
                        train_test_split(self.X, self.w, self.y, test_size=test_size, random_state=random_state)

        else:
            self.X_train, self.X_test, self.w_train, self.w_test, self.y_train, self.y_test, \
                _, self.mu_treat_test, _, self.mu_control_test, _, self.ite_test = \
                train_test_split(self.X, self.w, self.y, self.mu_treat, self.mu_control, self.ite,
                                 test_size=test_size, random_state=random_state)

    def fit(self) -> None:
        """Train IPMs."""
        for um in self.um_list:
            um.fit(self.X_train, self.y_train, self.w_train)

    def predict(self) -> None:
        """Estimate Individual Treatment Effects for test data."""
        # ite prediction
        self.ite_pred_list: list = [] if self.real_world else [self.ite_test]
        for um in self.um_list:
            if um.um_type == "ipm":
                self.ite_pred_list.append(um.predict(self.X_test))
            elif um.um_type == "tdf":
                self.ite_pred_list.append(um.predict_ite(self.X_test))

        # treatment allocation
        self.treatment_pred_list: list = [] if self.real_world else [np.array(self.ite_test > 0, dtype=int)]
        for um in self.um_list:
            if um.um_type == "ipm":
                self.treatment_pred_list.append(np.array(um.predict(self.X_test) > 0, dtype=int))
            elif um.um_type == "tdf":
                self.treatment_pred_list.append(um.predict_assignment(self.X_test))

    def aggregate_result(self, gamma: float=0.1) -> None:
        """Aggregate results of the experiment.

        Parameters
        ----------
        gamma: float, (default=0.1)
            The switching hyper-parameter gamma with which to estimate the uplift curve.

        """
        if self.rct:
            if not self.real_world:
                udf_list = [uplift_frame(y_obs=self.y_test, w=self.w_test, ite_pred=ite_pred,
                                         mu_treat=self.mu_treat_test, mu_control=self.mu_control_test,
                                         real_world=self.real_world, rct=self.rct, method=self.method) for ite_pred in self.ite_pred_list]

            elif not self.method == "naive":
                udf_list = [optimal_uplift_frame(y_obs=self.y_test, w=self.w_test, regression=self.regression)] + \
                           [uplift_frame(y_obs=self.y_test, w=self.w_test, ite_pred=ite_pred,
                                         mu_treat=self.mu_treat_test, mu_control=self.mu_control_test,
                                         real_world=self.real_world, rct=self.rct, method=self.method) for ite_pred in self.ite_pred_list]
            else:
                udf_list = [optimal_uplift_frame(y_obs=self.y_test, w=self.w_test, regression=self.regression)] + \
                           [uplift_frame(y_obs=self.y_test, w=self.w_test, ite_pred=ite_pred,
                                         real_world=self.real_world, rct=self.rct, method=self.method) for ite_pred in self.ite_pred_list]

        else:
            udf_list = [uplift_frame(y_obs=self.y_test, w=self.w_test, ite_pred=ite_pred, ps=self.ps_test,
                                     gamma=gamma, mu_treat=self.mu_treat_test, mu_control=self.mu_control_test,
                                     real_world=self.real_world, rct=self.rct, method=self.method) for ite_pred in self.ite_pred_list]

        self.lift_df_list, self.value_df_list = \
            [pd.concat([lift_df, _[["lift", "baseline_lift"]]]) for lift_df, _ in zip(self.lift_df_list, udf_list)], \
            [pd.concat([value_df, _[["value", "baseline_value"]]]) for value_df, _ in zip(self.value_df_list, udf_list)]

        self.auucs, self.aumucs = \
            [np.mean(udf.lift.values - udf.baseline_lift.values) for udf in udf_list], \
            [np.mean(udf.value.values - udf.baseline_value.values) for udf in udf_list]

        if not self.real_world or self.rct:
            self.auucs, self.aumucs = self.auucs / self.auucs[0], self.aumucs / self.aumucs[0]

            if not self.real_world:
                ite_list, value_list = \
                    [np.sqrt(mean_squared_error(self.ite_test, ite))
                     for ite in self.treatment_pred_list], \
                    [np.mean(self.mu_treat_test * d + self.mu_control_test * (1 - d))
                     for d in self.treatment_pred_list]
                _ite_df, _value_df = \
                    DataFrame({name: ite / ite_list[0] for name, ite in zip(self.name_list, ite_list)}, index=[self.num]), \
                    DataFrame({name: value / value_list[0] for name, value in zip(self.name_list, value_list)}, index=[self.num])
                self.ite_df, self.value_df = \
                    pd.concat([self.ite_df, _ite_df]), pd.concat([self.value_df, _value_df])

        # add results of evaluation metrics.
        _auuc_df, _aumuc_df = \
            DataFrame({name: auuc for name, auuc in zip(self.name_list, self.auucs)}, index=[self.num]), \
            DataFrame({name: value for name, value in zip(self.name_list, self.aumucs)}, index=[self.num])

        self.auuc_df, self.aumuc_df = \
            pd.concat([self.auuc_df, _auuc_df]), pd.concat([self.aumuc_df, _aumuc_df])

        # count the num of simulation.
        self.num += 1

    def run_sim(self, n_iter: int, gamma: float=0.1, test_size: float=0.5) -> None:
        """Run simulation.

        Parameters
        ----------
        n_iter: int
            Number of iterations.

        gamma: float, (default=0.1)
            The switching hyper-parameter gamma with which to estimate the uplift curve.

        test_size: float, (default=0.3)
            Represent the proportion of the dataset to include in the test split and should be between 0.0 and 1.0.

        """
        for i in np.arange(n_iter):
            self.split_data(test_size=test_size, random_state=i)
            self.fit()
            self.predict()
            self.aggregate_result(gamma=gamma)

    def output_curves(self) -> Tuple[Figure]:
        """Output the resulting uplift and the modified uplift curves.

        Returns
        -------
        fig_curve, fig_mcurve: Tuple[Figure]
            The resulting uplift and the modified uplift curveshe uplift curves.

        """
        lift_df_list, value_df_list = \
            [_df.mean(level=0) for _df in self.lift_df_list], \
            [_df.mean(level=0) for _df in self.value_df_list]
        fig_curve, fig_mcurve = \
            compare_uplift_curves(lift_df_list, self.name_list), \
            compare_modified_uplift_curves(value_df_list, self.name_list)

        return fig_curve, fig_mcurve

    def output_box(self) -> Tuple[Figure]:
        """Output the resulting box plots of AUUC and Max Values.

        Returns
        -------
        boxes: List[Figure]
            The resulting box plots of AUUC and Max Values.

        """
        auuc_boxes, aumuc_boxes = \
            [Box(y=self.auuc_df.loc[:, name].values, name=name) for name in self.name_list], \
            [Box(y=self.aumuc_df.loc[:, name].values, name=name) for name in self.name_list]
        fig_auuc, fig_aumuc = \
            Figure(data=auuc_boxes, layout=Layout(xaxis=dict(tickfont=dict(size=15)),
                                                  yaxis=dict(title="AUUCs", titlefont=dict(size=15)))), \
            Figure(data=aumuc_boxes, layout=Layout(xaxis=dict(tickfont=dict(size=15)),
                                                   yaxis=dict(title="AUMUCs", titlefont=dict(size=15))))

        if not self.real_world:
            value_boxes = [Box(y=self.value_df.loc[:, name].values, name=name) for name in self.name_list]
            fig_value = \
                Figure(data=value_boxes, layout=Layout(xaxis=dict(tickfont=dict(size=15)),
                                                       yaxis=dict(title="Values", titlefont=dict(size=15))))
            boxes = fig_auuc, fig_aumuc, fig_value
        else:
            boxes = fig_auuc, fig_aumuc

        return boxes
