"""This module contains tools for the model selection of uplift models."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, train_test_split

from metrics import sdr_mse, expected_response, uplift_frame
from optuna import create_study
from optuna.trial import Trial
from optuna.samplers import TPESampler


def cross_val_score(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                    mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
                    gamma: float=0.0, cv: int=3, scoring: str="value", random_state: Optional[int]=None) -> List[float]:
    """Evaluate metric(s) by cross-validation.

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : {array-like, sparse matrix} of shape = [n_samples, n_features]
        The training input samples. Sparse matrices are accepted only if
        they are supported by the base estimator.

    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).

    w : array-like, shape = [n_samples]
        The treatment assignment. The values should be binary.

    mu: array-like, shape = [n_samples], optional (default=None)
        The estimated potential outcomes.

    ps: array-like, shape = [n_samples], optional (default=None)
        The estimated propensity scores.

    gamma: float, optional (default=0.0)
        The switching hyper-parameter.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.

    scoring : string, optional, (default=None)
        A single string representing a evaluation metric.

    Returns
    -------
    scores : list of float
        Array of scores of the estimator for each run of the cross validation.

    """
    # Score list
    scores: list = []
    # KFold Splits
    kf = KFold(n_splits=cv, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        X_train, y_train, w_train = X[train_idx], y[train_idx], w[train_idx]
        X_test, y_test, w_test = X[test_idx], y[test_idx], w[test_idx]
        ps_test = ps[test_idx] if ps is not None else ps
        mu_test = mu[test_idx] if mu is not None else mu

        # ========== Fit ==========
        estimator.fit(X_train, y_train, w_train)
        # =========================

        # ========== Evaluate ==========
        if scoring == "mse":
            ite_pred = estimator.predict_ite(X_test)
            metric = sdr_mse(y=y_test, w=w_test, ite_pred=ite_pred,
                             mu=mu_test, ps=ps_test, gamma=gamma)

        elif scoring == "value":
            policy = estimator.predict(X_test)
            metric = expected_response(y=y_test, w=w_test, policy=policy, mu=mu_test, ps=ps_test)

        elif scoring in ["auuc", "aumuc"]:
            ite_pred, policy = estimator.predict_ite(X_test), estimator.predict(X_test)
            df = uplift_frame(ite_pred=ite_pred, policy=policy, y=y_test, w=w_test,
                              mu=mu_test, ps=ps_test, gamma=gamma, real_world=True)
            metric = np.mean(df.lift.values - df.baseline_lift.values) if scoring == "auuc" else np.mean(df.value.values)

        scores.append(metric)
        # ==============================

    return scores


class Objective:
    """Objective function for OptunaSearchCV class.

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    param_dist: Dict
        Dictionary with parameters names of meta-learning method (string) as keys and
        lists of parameter settings to try as values.

    X : {array-like, sparse matrix} of shape = [n_samples, n_features]
        The training input samples. Sparse matrices are accepted only if
        they are supported by the base estimator.

    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).

    w : array-like, shape = [n_samples]
        The treatment assignment. The values should be binary.

    mu: array-like, shape = [n_samples], optional (default=None)
        The estimated potential outcomes.

    ps: array-like, shape = [n_samples], optional (default=None)
        The estimated propensity scores.

    param_dist_base_model: Dict
        Dictionary with parameters names of base model (string) as keys and
        lists of parameter settings to try as values.

    param_dist_po_model: Dict
        Dictionary with parameters names of potential outcome model (string) as keys and
        lists of parameter settings to try as values.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.

    scoring : string, optional, (default=None)
        A single string representing a evaluation metric.

    """

    def __init__(self, estimator: BaseEstimator, param_dist: Dict,
                 X: np.ndarray, y: np.ndarray, w: np.ndarray,
                 mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
                 param_dist_base_model: Optional[dict]=None,
                 param_dist_po_model: Optional[dict]=None,
                 cv: int=3, scoring: str="value",
                 random_state: Optional[int]=None) -> None:
        """Initialize Class."""
        self.X = X
        self.y = y
        self.w = w
        self.mu = mu
        self.ps = ps
        self.estimator = estimator
        self.param_dist = param_dist
        self.param_dist_base_model = param_dist_base_model
        self.param_dist_po_model = param_dist_po_model
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

    def __call__(self, trial: Trial) -> float:
        """Callable."""
        estimator = clone(self.estimator)
        # set parameter distributions of the uplift modeling method.
        params = {
            name: trial._suggest(
                name, distribution
            ) for name, distribution in self.param_dist.items()
        }

        # set parameter distributions of the base model.
        if (self.param_dist_base_model is not None):
            base_model = clone(self.estimator.base_model)
            params_base_model = {
                name: trial._suggest(
                    name + "_base_model", distribution
                ) for name, distribution in self.param_dist_base_model.items()
            }

            base_model_ = base_model.set_params(**params_base_model)
            params.update({"base_model": base_model_})

        # set parameter distributions of the potential outcome model.
        if (self.param_dist_po_model is not None):
            po_model = clone(self.estimator.po_model)
            params_po_model = {
                name: trial._suggest(
                    name + "_po_model", distribution
                ) for name, distribution in self.param_dist_po_model.items()
            }

            po_model_ = po_model.set_params(**params_po_model)
            params.update({"po_model": po_model_})

        # set the parameters of the given uplift modeling method.
        estimator = self.estimator.set_params(**params)

        # estimate cross validation score.
        scores = cross_val_score(estimator=estimator,
                                 X=self.X, y=self.y, w=self.w, mu=self.mu, ps=self.ps,
                                 cv=self.cv, scoring=self.scoring, random_state=self.random_state)

        return np.mean(scores) if self.scoring == "mse" else - np.mean(scores)


class OptunaSearchCV(BaseEstimator):
    """OptunaSearchCV.

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    param_dist: Dict
        Dictionary with parameters names of meta-learning method (string) as keys and
        lists of parameter settings to try as values.

    param_dist_base_model: Dict
        Dictionary with parameters names of base model (string) as keys and
        lists of parameter settings to try as values.

    param_dist_po_model: Dict
        Dictionary with parameters names of potential outcome model (string) as keys and
        lists of parameter settings to try as values.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.

    scoring : string, (default=None)
        A single string representing a evaluation metric.

    n_iter: int, (default=10)
        The total number of search iterations.

    refit: bool, (default=True)
        Whether refit the best estimator or not.

    n_jobs: int, (default=1)
        The Number of parallel jobs.

    seed: int, optional, (default=None)

    random_state: int, optional, (default=None)

    """

    def __init__(self, estimator: BaseEstimator, param_dist: Dict,
                 param_dist_base_model: Optional[Dict]=None,
                 param_dist_po_model: Optional[Dict]=None,
                 cv: int=3, scoring: str="value",
                 n_iter: int=10, n_jobs: int=1, refit: bool=True,
                 seed: Optional[int]=None, random_state: Optional[int]=None) -> None:
        """Initialize Class."""
        self.cv = cv
        self.estimator = estimator
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_dist = param_dist
        self.param_dist_base_model = param_dist_base_model
        self.param_dist_po_model = param_dist_po_model
        self.refit = refit
        self.scoring = scoring
        self.seed = seed
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None) -> BaseEstimator:
        """Run fit with Tree-structured Parzen Estimator (TPE) Approach.

        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        w : array-like, shape = [n_samples]
            The treatment assignment. The values should be binary.

        mu: array-like, shape = [n_samples], optional (default=None)
            The estimated potential outcomes.

        ps: array-like, shape = [n_samples], optional (default=None)
            The estimated propensity scores.

        """
        objective = Objective(self.estimator, self.param_dist,
                              X, y, w, mu, ps,
                              self.param_dist_base_model, self.param_dist_po_model,
                              self.cv, self.scoring, self.random_state)
        self.sampler_ = TPESampler(seed=self.seed)
        self.study_ = create_study(sampler=self.sampler_)

        # run hyper-paramter optimization.
        self.study_.optimize(objective, n_jobs=self.n_jobs, n_trials=self.n_iter)

        # save the searching results.
        self.best_params_ = self.study_.best_params
        self.best_value_ = self.study_.best_value
        self.trials_dataframe = self.study_.trials_dataframe()
        # set the best estimator.
        estimator = clone(self.estimator)
        # best params of the uplift modeling method.
        params = {name: value
                  for name, value in self.best_params_.items()
                  if ("base_model" not in name) & ("po_model" not in name)}
        # best params of the base model.
        if (self.param_dist_base_model is not None):
            base_model = clone(self.estimator.base_model)
            params_base_model = {name.replace("_base_model", ""): value
                                 for name, value in self.best_params_.items()
                                 if "base_model" in name}
            params.update({"base_model": base_model.set_params(**params_base_model)})
        # best params of the potnetial outcome model.
        if (self.param_dist_po_model is not None):
            po_model = clone(self.estimator.po_model)
            params_po_model = {name.replace("_po_model", ""): value
                               for name, value in self.best_params_.items()
                               if "po_model" in name}
            params.update({"po_model": po_model.set_params(**params_po_model)})
        self.best_estimator_ = estimator.set_params(**params)

        # refit the best estimator to the given dataset.
        if self.refit:
            self.best_estimator_.fit(X, y, w)

        return self


def bootstrap_test_score(estimator: BaseEstimator,
                         X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
                         X_test: np.ndarray,  y_test: np.ndarray, w_test: np.ndarray,
                         mu_test: Optional[np.ndarray] = None, ps_test: Optional[np.ndarray] = None,
                         n_iter: int = 100) -> List[float]:
    """Compute the bootstrap metrics."""
    # ========== Fit ==========
    estimator.fit(X_train, y_train, w_train)
    # =========================

    # Compute the bootstrap scores.
    num_trts = np.unique(w_train).shape[0]
    mu_test = np.zeros(w_test.shape[0]) if mu_test is None else mu_test[np.arange(w_test.shape[0]), w_test]
    ps_test = pd.get_dummies(w_test).values.mean(axis=0)[w_test] if ps_test is None else ps_test[np.arange(w_test.shape[0]), w_test]
    values: list = []
    for i in np.arange(n_iter):
        # ========== predict for the bootstraps ==========
        np.random.seed(i)
        idx = np.random.choice(np.arange(X_test.shape[0]), size=X_test.shape[0], replace=True)
        X_boot, y_boot, w_boot, mu_boot, ps_boot = X_test[idx], y_test[idx], w_test[idx], mu_test, ps_test

        policy, _ = estimator.predict(X_boot)
        # ================================================

    # ========== Evaluate ==========
        indicator = np.array(w_boot == policy, dtype=int)
        value = np.mean(mu_boot + (y_boot - mu_boot) * indicator / ps_boot)
    # ==============================

        values.append(value)

    return values
