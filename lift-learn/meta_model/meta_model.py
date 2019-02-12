"""Classes for meta-learning methods."""
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils import check_array

from .base import UpliftModelInterface


class SMAClassifier(BaseEstimator, UpliftModelInterface):
    """Separate-Model Approach for Classification."""

    _uplift_model_type = "meta_model"

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
        self.pom = pom
        self.fitted_poms_: list = []
        self.name = f"SMA{name}" if name is not None else "SMA"

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Build an uplift model from the training set (X, y, w).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        w : array-like, shape = [n_samples]
            The treatment assignment.

        """
        n_trts = np.unique(w).shape[0]

        for trts_id in np.arange(n_trts):
            pom = clone(self.pom)
            pom.fit(X[w == trts_id], y[w == trts_id])
            self.fitted_poms_.append(pom)

    def predict(self, X: np.ndarray) -> None:
        """Predict optimal treatment for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The test input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        t : array of shape = [n_samples]
            The predicted optimal treatments.

        """
        pred_ite = self.predict_ite(X)
        _extented_pred_ite = np.concatenate([np.zeros((pred_ite.shape[0], i)), pred_ite], axis=1)
        return np.argmax(_extented_pred_ite, axis=1)

    def predict_ite(self, X: np.ndarray) -> None:
        """Predict individual treatment effects for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The test input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        ite : array of shape = [n_samples, (n_trts - 1)]
            The predicted individual treatment effects.

        """
        pred_ite = np.zeros((X.shape[0], len(self.fitted_poms_) - 1))
        pred_baseline = self.fitted_poms_[0].predict_proba(X)
        if pred_ite.ndim == 1:
            pred_ite = self.fitted_poms_[1].predict_proba(X) - pred_baseline
        else:
            for trts_id, model in enumerate(self.fitted_poms_[1:]):
                pred_ite[:, trts_id] = model.predict_proba(X) - pred_baseline

        return pred_ite


class SMARegressor(BaseEstimator, UpliftModelInterface):
    """Separate-Model Approach for Regression."""

    _uplift_model_type = "meta_model"

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
        self.pom = pom
        self.fitted_poms_: list = []
        self.name = f"SMA{name}" if name is not None else "SMA"

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Build an uplift model from the training set (X, y, w).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        w : array-like, shape = [n_samples]
            The treatment assignment.

        """
        n_trts = np.unique(w).shape[0]

        for trts_id in np.arange(n_trts):
            pom = clone(self.pom)
            pom.fit(X[w == trts_id], y[w == trts_id])
            self.fitted_poms_.append(pom)

    def predict(self, X: np.ndarray) -> None:
        """Predict optimal treatment for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The test input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        t : array of shape = [n_samples]
            The predicted optimal treatments.

        """
        pred_ite = self.predict_ite(X)
        _extented_pred_ite = np.concatenate([np.zeros((pred_ite.shape[0], i)), pred_ite], axis=1)
        return np.argmax(_extented_pred_ite, axis=1)

    def predict_ite(self, X: np.ndarray) -> None:
        """Predict individual treatment effects for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The test input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        ite : array of shape = [n_samples, (n_trts - 1)]
            The predicted individual treatment effects.

        """
        pred_ite = np.zeros((X.shape[0], len(self.fitted_poms_) - 1))
        pred_baseline = self.fitted_poms_[0].predict(X)
        if pred_ite.ndim == 1:
            pred_ite = self.fitted_poms_[1].predict(X) - pred_baseline
        for trts_id, model in enumerate(self.fitted_poms_[1:]):
            pred_ite[:, trts_id] = model.predict(X) - pred_baseline

        return pred_ite


class TOM(BaseEstimator, UpliftModelInterface):
    """Transformed Outcome Method for Regression and Classification.

    References
    ----------
    [1] Susan Athey and Guido W Imbens. Machine learning methods for estimating heterogeneous causal effects. arxiv, 2015.

    """

    _uplift_model_type = "meta_model"

    def __init__(self,
                 base_model: RegressorMixin,
                 ps_model: ClassifierMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        base_model: object
            The base model from which the IPM based on TOM is built.

        ps_model: object
            The predictive model for propensity score estimation.

        name: string, optional (default=None)
            The name of the model.

        """
        self.base_model = base_model
        self.ps_model = ps_model
        self.name = f"TOM({name})" if name is not None else "TOM"

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Build an uplift model from the training set (X, y, w).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        w : array-like, shape = [n_samples]
            The treatment assignment.

        """
        # estimate propensity scores.
        self.ps_model.fit(X, w)
        ps = self.ps_model.predict_proba(X)[:, 1]

        # fit the base model.
        transformed_outcome = self._transform_outcome(y, w, ps)
        self.base_model.fit(X, transformed_outcome)

    def predict(self, X: np.ndarray) -> None:
        """Predict optimal treatment for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The test input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        t : array of shape = [n_samples]
            The predicted optimal treatments.

        """
        pred_ite = self.predict_ite(X)
        return np.array(pred_ite > 0, dtype=int)

    def predict_ite(self, X: np.ndarray) -> None:
        """Predict individual treatment effects for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The test input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        ite : array of shape = [n_samples, (n_trts - 1)]
            The predicted individual treatment effects.

        """
        return self.base_model.predict(X)

    def _transform_outcome(self, y: np.ndarray, w: np.ndarray, ps: np.ndarray) -> np.ndarray:
        """Calcurate Transformed Outcomes.

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        w : array-like, shape = [n_samples]
            The treatment assignment.

        ps: array-like, shape = [n_samples]
            The estimated propensity scores.

        Returns
        ----------
        to: array-like, shape = [n_samples]
            The transformed outcomes.

        """
        return w * y / ps - (1 - w) * y / (1. - ps)


class CVT(BaseEstimator, UpliftModelInterface):
    """Class Variable Transformation for Classification.

    References
    ----------
    [1] Maciej Jaskowski and Szymon Jaroszewicz. Uplift modeling for clinical trial data. In ICML Workshop on Clinical Data Analysis, 2012.

    """

    _uplift_model_type = "meta_model"

    def __init__(self,
                 base_model: ClassifierMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class.

        Parameters
        ----------
        base_model: object
            The base model from which the IPM based on CVT is built.

        name: string, optional (default=None)
            The name of the model.

        """
        self.base_model = base_model
        self.name = f"CVT({name})" if name is not None else "CVT"

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Build an uplift model from the training set (X, y, w).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        w : array-like, shape = [n_samples]
            The treatment assignment.

        """
        # fit the base model.
        transformed_outcome = w * y + (1 - w) * (1 - y)
        self.base_model.fit(X, transformed_outcome)

    def predict(self, X: np.ndarray) -> None:
        """Predict optimal treatment for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The test input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        t : array of shape = [n_samples]
            The predicted optimal treatments.

        """
        pred_ite = self.predict_ite(X)
        return np.array(pred_ite > 0, dtype=int)

    def predict_ite(self, X: np.ndarray) -> None:
        """Predict individual treatment effects for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The test input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        ite : array of shape = [n_samples, (n_trts - 1)]
            The predicted individual treatment effects.

        """
        return self.base_model.predict_proba(X)


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
