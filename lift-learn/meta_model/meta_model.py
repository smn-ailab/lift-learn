"""Classes for meta-learning methods."""
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression as LR
from sklearn.utils import check_array

from .base import UpliftModelInterface


class SMAClassifier(BaseSMA):
    """Separate-Model Approach for Classification."""

    def __init__(self,
                 pom: ClassifierMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        pom: object
            The Potential Outcome Model from which the TDF based on SMA is built.

        name: string, optional (default=None)
            The name of the model.

        """
        if not isinstance(pom, ClassifierMixin):
            raise TypeError("set Classifier as pom.")

        super().__init__(pom, "classification", name)


class SMARegressor(BaseSMA):
    """Separate-Model Approach for Regression."""

    def __init__(self,
                 pom: RegressorMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        pom: object
            The Potential Outcome Model from which the TDF based on SMA is built.

        name: string, optional (default=None)
            The name of the model.

        """
        if not isinstance(pom, RegressorMixin):
            raise TypeError("set Regressor as pom.")

        super().__init__(pom, "regression", name)


class TOM(BaseTOM):
    """Transformed Outcome Method for Regression and Classification.

    References
    ----------
    [1] Susan Athey and Guido W Imbens. Machine learning methods for estimating heterogeneous causal effects. arxiv, 2015.

    """

    def __init__(self,
                 base_model: RegressorMixin,
                 ps_estimator: ClassifierMixin=LR(C=100, solver="lbfgs", random_state=0),
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        base_model: object
            The base model from which the IPM based on TOM is built.

        name: string, optional (default=None)
            The name of the model.

        """
        if not isinstance(base_model, RegressorMixin):
            raise TypeError("set Regressor as base_model.")
        if not isinstance(ps_estimator, ClassifierMixin):
            raise TypeError("set Classifier as ps_estimator.")

        super().__init__(base_model, ps_estimator, name)

    def fit(self, X: np.ndarray, y_obs: np.ndarray, w: np.ndarray) -> None:
        """Build a Transformed Outcome Regressor from the training set (X, y_obs, w).

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Training input samples.

        y_obs : array-like of shape = (n_samples)
            Observed target values.(class labels in classification, real numbers in regression).

        w : array-like of shape = (n_samples)
            Treatment assignment indicators.

        ps: array-like of shape = (n_samples)
            Estimated propensity scores.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])
        if not isinstance(y_obs, np.ndarray):
            raise TypeError("y_obs must be a numpy.ndarray.")
        if not isinstance(w, np.ndarray):
            raise TypeError("w must be a numpy.ndarray.")

        # estimate the propensity score.
        self.ps_estimator.fit(X, w)
        ps = self.ps_estimator.predict_proba(X)[:, 1]
        transformed_outcome = self._calc_transformed_outcome(y_obs, w, ps)
        self.base_model.fit(X, transformed_outcome)


class CVT(BaseTOM):
    """Class Variable Transformation for Classification.

    References
    ----------
    [1] Maciej Jaskowski and Szymon Jaroszewicz. Uplift modeling for clinical trial data. In ICML Workshop on Clinical Data Analysis, 2012.

    """

    def __init__(self,
                 base_model: ClassifierMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        base_model: object
            The base estimator from which the IPM based on CVT is built.

        name: string, optional (default=None)
            The name of the model.

        """
        if not isinstance(base_model, ClassifierMixin):
            raise TypeError("set Classifier as base_model.")

        super().__init__(base_model, None, name)

    def fit(self, X: np.ndarray, y_obs: np.ndarray, w: np.ndarray) -> None:
        """Build a CVT estimator from the training set (X, y_obs, w).

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Training input samples.

        y_obs : array-like of shape = (n_samples)
            Observed target values.(class labels).

        w : array-like of shape = (n_samples)
            Treatment assignment indicators.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])
        if not isinstance(y_obs, np.ndarray):
            raise TypeError("y_obs must be a numpy.ndarray.")
        if not isinstance(w, np.ndarray):
            raise TypeError("w must be a numpy.ndarray.")

        z = y_obs * w + (1 - y_obs) * (1 - w)
        self.base_model.fit(X, z)


class SDRMClassifier(BaseSDRM):
    """Switch Doubly Robust Method for Classification.

    References
    ----------
    [1] Doubly Robust Prediction and Evaluation Methods Improve Uplift Modeling for Observaional Data, 2018.

    """

    def __init__(self,
                 base_model: RegressorMixin,
                 pom_treat: ClassifierMixin,
                 pom_control: ClassifierMixin,
                 ps_estimator: ClassifierMixin=LR(C=100, solver="lbfgs", random_state=0),
                 gamma: float=0.0,
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        base_model: object
            The base model from which the IPM based on SDRM is built.

        pom_treat: object
            The Potential Outcome Model for the treated from which the proxy_outcome is calculated.

        pom_control: object
            The Potential Outcome Model for the controlled from which the proxy_outcome is calculated.

        gamma: float, (default=0,0)
            The switching hyper-parameter gamma for SDRM.

        name: string, optional (default=None)
            The name of the model.

        """
        if not isinstance(pom_treat, ClassifierMixin):
            raise TypeError("set Classifier as pom_treat.")
        if not isinstance(pom_control, ClassifierMixin):
            raise TypeError("set Classifier as pom_control.")

        BaseSDRM.__init__(self, base_model, pom_treat, pom_control, ps_estimator, gamma, name)

    def fit(self, X: np.ndarray, y_obs: np.ndarray, w: np.ndarray) -> None:
        """Build a SDRM Classifier from the training set (X, y_obs, w).

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            Training input samples.

        y_obs : array-like of shape = (n_samples)
            Observed target values.(class labels).

        w : array-like of shape = (n_samples)
            Treatment assignment indicators.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=[int, float])

        super().fit(X, y_obs, w)
        pred_treat, pred_control = self.pom_treat.predict_proba(X)[:, 1], self.pom_control.predict_proba(X)[:, 1]
        proxy_outcome = self._calc_switch_doubly_robust_outcome(y_obs, w, self.ps,
                                                                pred_treat, pred_control, self.gamma)
        self.base_model.fit(X, proxy_outcome)


class SDRMRegressor(BaseSDRM):
    """Switch Doubly Robust Method for Regression.

    References
    ----------
    [1] Doubly Robust Prediction and Evaluation Methods Improve Uplift Modeling for Observaional Data, 2018.

    """

    def __init__(self,
                 base_model: RegressorMixin,
                 pom_treat: RegressorMixin,
                 pom_control: RegressorMixin,
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

        gamma: float, (default=0,0)
            The switching hyper-parameter gamma for SDRM.

        name: string, optional (default=None)
            The name of the model.

        """
        if not isinstance(pom_treat, RegressorMixin):
            raise TypeError("set Regressor as pom_treat.")
        if not isinstance(pom_control, RegressorMixin):
            raise TypeError("set Regressor as pom_control.")

        BaseSDRM.__init__(self, base_model, pom_treat, pom_control, ps_estimator, gamma, name)

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

        super().fit(X, y_obs, w)
        pred_treat, pred_control = self.pom_treat.predict(X), self.pom_control.predict(X)
        proxy_outcome = self._calc_switch_doubly_robust_outcome(y_obs, w, self.ps,
                                                                pred_treat, pred_control, self.gamma)
        self.base_model.fit(X, proxy_outcome)
