"""Base classes for meta-learning methods."""
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import (BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone,
                          RegressorMixin)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.utils import check_array


class BaseTMA(BaseEstimator, MetaEstimatorMixin):
    """Base for all Two-Model Approach classes."""

    # Two-Model Approach is an ITE Prediction Model.
    um_type = "ipm"

    def __init__(self,
                 pom_treat: Union[ClassifierMixin, RegressorMixin],
                 pom_control: Union[ClassifierMixin, RegressorMixin],
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        pom_treat: object
            The Potential Outcome Model for the treated from which the IPM based on TMA  is built.

        pom_control: object
            The Potential Outcome Model for the controlled from which the IPM based on TMA is built.

        name: string, optional (default=None)
            The name of the model.

        """
        self.pom_treat, self.pom_control = pom_treat, pom_control
        self.name = f"TMA({name})" if name is not None else "TMA"

    def fit(self, X: np.ndarray, y_obs: np.ndarray, w: np.ndarray) -> None:
        """Build a Two-Model Estimator from the training set (X, y_obs, w).

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Training input samples.

        y_obs : array-like of shape = (n_samples)
            Observed target values. (class labels in classification, real numbers in
            regression).

        w : array-like of shape = (n_samples)
            Treatment assignment indicators.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])
        if not isinstance(y_obs, np.ndarray):
            raise TypeError("y_obs must be a numpy.ndarray.")
        if not isinstance(w, np.ndarray):
            raise TypeError("w must be a numpy.ndarray.")

        # set features and target values.
        X_treat, X_control, y_treat, y_control = X[w == 1], X[w == 0], y_obs[w == 1], y_obs[w == 0]

        # fit estimators for both treatment and control groups.
        self.pom_treat.fit(X_treat, y_treat)
        self.pom_control.fit(X_control, y_control)


class BaseSMA(BaseEstimator, MetaEstimatorMixin):
    """Base for all Separate-Model Approach classes."""

    # Separate-Model Approach is a Treatment Decicion Function.
    um_type = "tdf"

    def __init__(self,
                 pom: Union[ClassifierMixin, RegressorMixin],
                 task: str,
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        pom: object
            The Potential Outcome Model from which the IPM based on TMA  is built.

        task: str
            Regression or Classification.

        name: string, optional (default=None)
            The name of the base model.

        """
        self.pom = pom
        self.task = task
        self.name = f"SMA({name})" if name is not None else "SMA"

    def fit(self, X: np.ndarray, y_obs: np.ndarray, w: np.ndarray) -> None:
        """Build a Separate-Model Estimator from the training set (X, y_obs, w).

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Training input samples.

        y_obs : array-like of shape = (n_samples)
            Observed target values. (class labels in classification, real numbers in
            regression).

        w : array-like of shape = (n_samples)
            Treatment assignment indicators.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])
        if not isinstance(y_obs, np.ndarray):
            raise TypeError("y_obs must be a numpy.ndarray.")
        if not isinstance(w, np.ndarray):
            raise TypeError("w must be a numpy.ndarray.")

        self.num_trts = np.unique(w).shape[0]
        # fit base models
        self.pom_list: list = []
        for t in np.arange(self.num_trts):
            pom = clone(self.pom)
            pom.fit(X[w == t], y_obs[w == t])
            self.pom_list.append(pom)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the Individual Treatmenf Effects.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Test input samples.

        Returns
        -------
        ite_pred  : array-like of shape = (n_samples)
            Estimated Individual Treatment Effects of the input samples.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])

        value_pred = np.empty((X.shape[0], self.num_trts))
        # Let each tree make a prediction on the data
        for trts_id, pom in enumerate(self.pom_list):
            value_pred[:, trts_id] = pom.predict_proba(X)[:, 1] if self.task == "classification" else pom.predict(X)
        ite_pred = value_pred - np.expand_dims(value_pred[:, 0], axis=1)
        policy = np.argmax(ite_pred, axis=1)

        return policy, ite_pred

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        """Predict the Individual Treatmenf Effects.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Test input samples.

        Returns
        -------
        ite_pred  : array-like of shape = (n_samples)
            Estimated Individual Treatment Effects of the input samples.

        """
        ite_pred = self.predict(X)[1]

        return ite_pred

    def predict_assignment(self, X: np.ndarray) -> np.ndarray:
        """Predict the Individual Treatmenf Effects.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Test input samples.

        Returns
        -------
        ite_pred  : array-like of shape = (n_samples)
            Estimated Individual Treatment Effects of the input samples.

        """
        policy = self.predict(X)[0]

        return policy


class BaseTOM(BaseEstimator, MetaEstimatorMixin):
    """Base for all Transformed Outcome Method classes."""

    # Transformed Outcome Method is a ITE Prediction Model.
    um_type = "ipm"

    def __init__(self,
                 base_model: Union[ClassifierMixin, RegressorMixin],
                 ps_estimator: ClassifierMixin=LR(C=100, solver="lbfgs", random_state=0),
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        base_model: object
            The base model from which the IPM based on TOM is built.

        ps_estimator: object (default=LogisticRegression(C=100, solver="lbfgs", random_state=0))
            The Propensity Score estimator.

        name: string, optional (default=None)
            The name of the model.

        """
        self.base_model, self.ps_estimator = base_model, ps_estimator
        self.name = f"TOM({name})" if name is not None else "TOM"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the Individual Treatmenf Effects.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Test input samples.

        Returns
        -------
        ite_pred  : array-like of shape = (n_samples)
            Estimated Individual Treatment Effects of the input samples.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])

        ite_pred = self.base_model.predict(X)
        return ite_pred

    def _calc_transformed_outcome(self, y_obs: np.ndarray, w: np.ndarray, ps: np.ndarray) -> np.ndarray:
        """Calculate Transformed Outcome.

        Parameters
        ----------
        y_obs : array-like of shape = (n_samples)
            Observed target values.(class labels in classification, real numbers in regression).

        w : array-like of shape = (n_samples)
            Treatment assignment indicators.

        ps: array-like of shape = (n_samples)
            Estimated propensity scores.

        Returns
        -------
        transformed_outcome: array-like of shape = (n_samples)

        """
        transformed_outcome = (w * y_obs / ps) - ((1.0 - w) * y_obs / (1.0 - ps))
        return transformed_outcome


class BaseSDRM(BaseEstimator, MetaEstimatorMixin):
    """Base for all Switch Doubly Robust Method classes."""

    # Switch Doubly Robust Method is a ITE Prediction Model.
    um_type = "ipm"

    def __init__(self,
                 base_model: Union[ClassifierMixin, RegressorMixin],
                 pom_treat: Union[ClassifierMixin, RegressorMixin],
                 pom_control: Union[ClassifierMixin, RegressorMixin],
                 ps_estimator: ClassifierMixin=LR(C=100, solver="lbfgs", random_state=0),
                 gamma: float=0.0,
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        base_model: object
            The base model from which the IPM based on SDRM is built.

        pom_treat: object
            The Potential Outcome Model for the treated from which the proxy outcome is calculated.

        pom_control: object
            The Potential Outcome Model for the controlled from which the proxy outcome is calculated.

        ps_estimator: object (default=LogisticRegression(C=100, solver="lbfgs", random_state=0))
            The Propensity Score estimator.

        name: string, optional (default=None)
            The name of the model.

        """
        if not isinstance(base_model, RegressorMixin):
            raise TypeError("set Regressor as base_model.")
        if not isinstance(ps_estimator, ClassifierMixin):
            raise TypeError("set Classifier as ps_estimator.")
        if not isinstance(gamma, float):
            raise TypeError("gamma must be a float.")
        assert (gamma >= 0.0) and (gamma <= 1), "gamma must be between 0 and 1."

        self.base_model, self.pom_treat, self.pom_control, self.ps_estimator, self.gamma = \
            base_model, pom_treat, pom_control, ps_estimator, gamma
        self.name = f"SDRM({name})" if name is not None else "SDRM"

    def fit(self, X: np.ndarray, y_obs: np.ndarray, w: np.ndarray) -> None:
        """Build a SDRM Regressor from the training set (X, y_obs, w).

        :param X  : array-like of shape = (n_samples, n_features)
                    The training input samples.

        :param y_obs  : array-like of shape = (n_samples)
                    The target values.(real numbers).

        :param w  : array-like of shape = (n_samples)
                    The treatment assignment values.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])

        # estimate the propensity score.
        self.ps_estimator.fit(X, w)
        self.ps = self.ps_estimator.predict_proba(X)[:, 1]

        # set features and target values.
        X_treat, X_control, y_treat, y_control = X[w == 1], X[w == 0], y_obs[w == 1], y_obs[w == 0]

        # fit estimators for both treatment and control groups.
        self.pom_treat.fit(X_treat, y_treat)
        self.pom_control.fit(X_control, y_control)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the Individual Treatmenf Effects.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Test input samples.

        Returns
        -------
        ite_pred  : array-like of shape = (n_samples)
            Estimated Individual Treatment Effects of the input samples.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])

        ite_pred = self.base_model.predict(X)
        return ite_pred

    def _calc_switch_doubly_robust_outcome(self, y_obs: np.ndarray, w: np.ndarray, ps: np.ndarray,
                                           mu_treat: np.ndarray, mu_control: np.ndarray, gamma: float=0.0) -> np.ndarray:
        """Calculate Switch Doubly Robust Transformed Outcome.

        Parameters
        ----------
        y : array-like of shape = (n_samples)
            Observed target values.(real numbers).

        w : array-like of shape = (n_samples)
            Treatment assignment indicators.

        ps: array-like of shape = (n_samples)
            Estimated propensity scores.

        mu_treat: array-like of shape = (n_samples), optional
            Estimated potential outcomes for the treated.

        mu_control: array-like of shape = (n_samples), optional
            Estimated potential outcomes for the untreated.

        gamma: float, (default=0,0)
            The switching hyper-parameter gamma for SDRM.

        Returns
        -------
        sdr_transformed_outcome  : array-like of shape = (n_samples)
            The switch doubly robust transformed outcome.

        """
        if not isinstance(y_obs, np.ndarray):
            raise TypeError("y_obs must be a numpy.ndarray.")
        if not isinstance(w, np.ndarray):
            raise TypeError("w must be a numpy.ndarray.")
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
        sdr_transformed_outcome = ((w * (y_obs - mu_treat)) / ps) - (((1 - w) * (y_obs - mu_control)) / (1 - ps)) + (mu_treat - mu_control)
        sdr_transformed_outcome[(w == 1) & (ps < gamma)] = direct_method[(w == 1) & (ps < gamma)]
        sdr_transformed_outcome[(w == 0) & (ps > 1 - gamma)] = direct_method[(w == 0) & (ps > 1 - gamma)]
        return sdr_transformed_outcome
