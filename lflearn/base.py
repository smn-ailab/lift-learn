"""Base classes for all uplift models."""
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


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
        pred_ite = self.predict_ite(X)
        _extented_pred_ite = np.concatenate([np.zeros((pred_ite.shape[0], 1)), pred_ite], axis=1)
        return np.argmax(_extented_pred_ite, axis=1)

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


class PropensityBasedModel(BaseEstimator, UpliftModelInterface):
    """Base class for all propensity based uplift models in lift-learn."""

    def __init__(self,
                 base_model: Union[ClassifierMixin, RegressorMixin],
                 ps_model: Optional[ClassifierMixin]=None) -> None:
        """Initialize Class."""
        self.base_model = base_model
        self.ps_model = ps_model

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

    def _transform_outcome(y: np.ndarray, w: np.ndarray, ps: np.ndarray,
                           mu: Optional[np.ndarray]=None, gamma: float=0.0) -> np.ndarray:
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
