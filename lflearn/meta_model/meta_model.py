"""Classes for meta-learning methods."""
from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import check_array

from base import MultiTreatmentError, SDRMCommon, SMACommon, TransformationBasedModel


class SMAClassifier(SMACommon):
    """Separate-Model Approach for Classification.

    Parameters
    ----------
    po_model: object
        The predictive model for potential outcome estimation.

    name: string, optional (default=None)
        The name of the model.

    """

    _uplift_model_type = "meta_model"

    def __init__(self,
                 po_model: ClassifierMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class."""
        super().__init__(po_model, name, True)


class SMARegressor(SMACommon):
    """Separate-Model Approach for Classification.

    Parameters
    ----------
    po_model: object
        The predictive model for potential outcome estimation.

    name: string, optional (default=None)
        The name of the model.

    """

    _uplift_model_type = "meta_model"

    def __init__(self,
                 po_model: RegressorMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class."""
        super().__init__(po_model, name, False)


class TOM(TransformationBasedModel):
    """Transformed Outcome Method for Regression and Classification.

    Parameters
    ----------
    base_model: object
        The base model from which TOM is built.

    ps_model: object
        The predictive model for propensity score estimation.

    name: string, optional (default=None)
        The name of the model.

    References
    ----------
    [1] Susan Athey and Guido W Imbens.  "Machine learning methods for
        estimating heterogeneous causal effects", arxiv, 2015.

    """

    _uplift_model_type = "meta_model"

    def __init__(self,
                 base_model: RegressorMixin,
                 ps_model: ClassifierMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class."""
        super().__init__(base_model, ps_model)
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
            The treatment assignment. The values should be binary.

        """
        if np.unique(w).shape[0] != 2:
            raise MultiTreatmentError("treatment assignments shoul be binary values")

        # estimate propensity scores.
        self.ps_model.fit(X, w)
        ps = self.ps_model.predict_proba(X)

        # fit the base model.
        transformed_outcome = self._transform_outcome(y, w, ps)
        self.base_model.fit(X, transformed_outcome)


class CVT(TransformationBasedModel):
    """Class Variable Transformation for Classification.

    Parameters
    ----------
    base_model: object
        The base model from which CVT is bulit.

    name: string, optional (default=None)
        The name of the model.

    References
    ----------
    [1] Maciej Jaskowski and Szymon Jaroszewicz. "Uplift modeling for
        clinical trial data", In ICML Workshop on Clinical Data Analysis, 2012.

    """

    _uplift_model_type = "meta_model"

    def __init__(self,
                 base_model: ClassifierMixin,
                 name: Optional[str]=None) -> None:
        """Initialize Class."""
        super().__init__(base_model)
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
            The treatment assignment. The values should be binary.

        """
        if np.unique(w).shape[0] != 2:
            raise MultiTreatmentError("treatment assignments shoul be binary values")

        # fit the base model.
        transformed_outcome = w * y + (1 - w) * (1 - y)
        self.base_model.fit(X, transformed_outcome)


class SDRMClassifier(SDRMCommon):
    """Switch Doubly Robust Method for Classification.

    Parameters
    ----------
    base_model: object
        The base model from which the SDRM is built.

    po_model: object
        The predictive model for potential outcome estimation.

    ps_model: object
        The predictive model for propensity score estimation.

    gamma: float, optional (default=0,0)
        The switching hyper-parameter.

    name: string, optional (default=None)
        The name of the model.

    References
    ----------
    [1] Yuta Saito, Hayato Sakata, and Kazuhide Nakata. "Doubly Robust Prediction
        and Evaluation Methods Improve Uplift Modeling for Observaional Data",
        In Proceedings of the SIAM International Conference on Data Mining, 2019.

    """

    _uplift_model_type = "meta_model"

    def __init__(self,
                 base_model: RegressorMixin,
                 po_model: ClassifierMixin,
                 ps_model: ClassifierMixin,
                 gamma: float=0.0,
                 name: Optional[str]=None) -> None:
        """Initialize Class."""
        super().__init__(base_model, po_model, ps_model, gamma, name, True)


class SDRMRegressor(SDRMCommon):
    """Switch Doubly Robust Method for Classification.

    Parameters
    ----------
    base_model: object
        The base model from which the SDRM is built.

    po_model: object
        The predictive model for potential outcome estimation.

    ps_model: object
        The predictive model for propensity score estimation.

    gamma: float, optional (default=0,0)
        The switching hyper-parameter.

    name: string, optional (default=None)
        The name of the model.

    References
    ----------
    [1] Yuta Saito, Hayato Sakata, and Kazuhide Nakata. "Doubly Robust Prediction
        and Evaluation Methods Improve Uplift Modeling for Observaional Data",
        In Proceedings of the SIAM International Conference on Data Mining, 2019.

    """

    _uplift_model_type = "meta_model"

    def __init__(self,
                 base_model: RegressorMixin,
                 po_model: RegressorMixin,
                 ps_model: ClassifierMixin,
                 gamma: float=0.0,
                 name: Optional[str]=None) -> None:
        """Initialize Class."""
        super().__init__(base_model, po_model, ps_model, gamma, name, False)
