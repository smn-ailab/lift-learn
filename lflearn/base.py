"""Base classes for all uplift models."""
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from exceptions import MultiTreatmentError, NotFittedError


class UpliftModelInterface:
    """Abstract base class for all uplift models in lift-learn."""

    @abstractmethod
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
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
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
        pass

    @abstractmethod
    def predict_ite(self, X: np.ndarray) -> np.ndarray:
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
        pass


class TransformationBasedModel(BaseEstimator, UpliftModelInterface):
    """Base class for all transformation based uplift models in lift-learn."""

    def __init__(self,
                 base_model: Union[ClassifierMixin, RegressorMixin],
                 ps_model: Optional[ClassifierMixin]=None) -> None:
        """Initialize Class."""
        self.base_model = base_model
        if ps_model is not None:
            self.ps_model = ps_model

    def predict(self, X: np.ndarray) -> np.ndarray:
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

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
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

    def _transform_outcome(self, y: np.ndarray, w: np.ndarray, ps: np.ndarray,
                           mu: Optional[np.ndarray]=None, gamma: float=0.0) -> np.ndarray:
        """Calcurate Transformed Outcomes.

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        w : array-like, shape = [n_samples]
            The treatment assignment. The values should be binary.

        ps: array-like, shape = [n_samples]
            The estimated propensity scores.

        mu: array-like, shape = [n_samples]
            The estimated potential outcomes.

        gamma: float, optional (default=0.0)

        Returns
        ----------
        transformed_outcome: array-like, shape = [n_samples]
            The transformed outcomes.

        """
        mu = np.zeros((y.shape[0], 2)) if mu is None else mu

        direct_estimates = mu[:, 1] - mu[:, 0]
        transformed_outcome = w * (y - mu[:, 1]) / ps[:, 1] - (1 - w) * (y - mu[:, 0]) / ps[:, 0] + direct_estimates
        transformed_outcome[(w == 1) & (ps[:, 1] < gamma)] = direct_estimates[(w == 1) & (ps[:, 1] < gamma)]
        transformed_outcome[(w == 0) & (ps[:, 0] < gamma)] = direct_estimates[(w == 0) & (ps[:, 0] < gamma)]
        return transformed_outcome


class SMACommon(BaseEstimator, UpliftModelInterface):
    """Base Class for SMA."""

    def __init__(self,
                 po_model: RegressorMixin,
                 name: Optional[str]=None,
                 is_classifier: bool=True) -> None:
        """Initialize Class."""
        self.po_model = po_model
        self.fitted_po_models: list = []
        self.name = f"SMA{name}" if name is not None else "SMA"
        self.is_classifier = is_classifier

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
            po_model = clone(self.po_model)
            po_model.fit(X[w == trts_id], y[w == trts_id])
            self.fitted_po_models.append(po_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        # Adding all zero value columns to choose the best treatment including baseline one.
        _extented_pred_ite = np.concatenate([np.zeros((pred_ite.shape[0], 1)), pred_ite], axis=1)
        return np.argmax(_extented_pred_ite, axis=1)

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
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
        if self.fitted_po_models == []:
            raise NotFittedError("Call fit before prediction")

        pred_ite = np.zeros((X.shape[0], len(self.fitted_po_models) - 1))
        baselines: list = []
        for trts_id, model in enumerate(self.fitted_po_models):
            ites = model.predict_proba(X)[:, 1] if self.is_classifier else model.predict(X)
            if trts_id == 0:
                baselines = ites
            else:
                pred_ite[:, trts_id - 1] = ites - baselines

        return pred_ite


class SDRMCommon(TransformationBasedModel):
    """Base Class for SDRM."""

    def __init__(self,
                 base_model: RegressorMixin,
                 po_model: Union[ClassifierMixin, RegressorMixin],
                 ps_model: ClassifierMixin,
                 gamma: float=0.0,
                 name: Optional[str]=None,
                 is_classifier: bool=True) -> None:
        """Initialize Class."""
        super().__init__(base_model, ps_model)
        self.po_model = po_model
        self.fitted_po_models: list = []
        self.gamma = gamma
        self.name = f"SDRM({name})" if name is not None else "SDRM"
        self.is_classifier = is_classifier

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
            The treatment assignment. The values should be binary.

        """
        if np.unique(w).shape[0] != 2:
            raise MultiTreatmentError("treatment assignments should be binary values")

        # estimate propensity scores.
        self.ps_model.fit(X, w)
        ps = self.ps_model.predict_proba(X)

        # estimate potential outcomes.
        estimated_potential_outcomes = np.zeros((X.shape[0], 2))
        for trts_id in np.arange(2):
            po_model = clone(self.po_model)
            po_model.fit(X[w == trts_id], y[w == trts_id])
            self.fitted_po_models.append(po_model)
            estimated_potential_outcomes[:, trts_id] = po_model.predict_proba(X)[:, 1] if self.is_classifier else po_model.predict(X)

        # fit the base model.
        transformed_outcome = self._transform_outcome(y, w, ps, estimated_potential_outcomes, self.gamma)
        self.base_model.fit(X, transformed_outcome)
