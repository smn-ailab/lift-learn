"""This module contains tools for the model selection of uplift models."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, train_test_split

from causalml.meta import SMAClassifier, SMARegressor
from causalml.metrics import cab_mse, expected_response, uplift_frame
from optuna import create_study
from optuna.samplers import TPESampler


def cross_val_score(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                    mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
                    gamma: float=0.0, cv: int=3, scoring: str="value", random_state: int=0) -> List[float]:
    """Cross valudation for uplift modeling."""
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

        # ========== Predict ==========
        if estimator.um_type == "ipm":
            ite_pred = estimator.predict(X_test)
            policy = np.array(ite_pred > 0, dtype=int)
        elif estimator.um_type == "tdf":
            policy, ite_pred = estimator.predict(X_test)
        # =============================

        # ========== Evaluate ==========
        if scoring == "mse":
            metric = cab_mse(y_test, w_test, ite_pred, mu_test, ps_test, gamma)

        elif scoring == "value":
            metric = expected_response(y_test, w_test, policy, mu_test, ps_test)

        elif scoring in ["auuc", "aumuc"]:
            df = uplift_frame(ite_pred=ite_pred, y=y_test, w=w_test,
                              mu=mu_test, ps=ps_test, gamma=gamma, real_world=True)
            metric = np.mean(df.lift.values - df.baseline_lift.values) if scoring == "auuc" else np.mean(df.value.values)

        scores.append(metric)
        # ==============================

    return scores


class Objective:
    """Objective function."""

    def __init__(self, estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                 mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None,
                 param_dist: Optional[dict]=None, pom: Optional[BaseEstimator]=None, param_dist_pom: Optional[dict]=None,
                 cv: int=3, scoring: str="value") -> None:
        """Initialize Class."""
        self.X = X
        self.y = y
        self.w = w
        self.mu = mu
        self.ps = ps
        self.estimator = estimator
        self.pom = pom
        self.param_dist = param_dist
        self.param_dist_pom = param_dist_pom
        self.cv = cv
        self.scoring = scoring

    def __call__(self, trial):
        """Callable."""
        if self.param_dist is not None:
            params = {
                name: trial._suggest(
                    name, distribution
                ) for name, distribution in self.param_dist.items()
            }

        else:
            params = {}

        if (self.pom is not None) and (self.param_dist_pom is not None):
            params_pom = {
                name: trial._suggest(
                    name + "_pom", distribution
                ) for name, distribution in self.param_dist_pom.items()
            }

            pom = self.pom(**params_pom)
            params.update({"pom": pom})

        estimator = self.estimator(**params)

        scores = cross_val_score(estimator=estimator,
                                 X=self.X, y=self.y, w=self.w, mu=self.mu, ps=self.ps,
                                 cv=self.cv, scoring=self.scoring, random_state=0)

        return - np.mean(scores)


class OptunaSearchCV(BaseEstimator):
    """Class for optuna search."""

    def __init__(self, estimator: BaseEstimator,
                 param_dist: Optional[dict]=None, pom: Optional[BaseEstimator]=None, param_dist_pom: Optional[dict]=None,
                 cv: int=3, scoring: str="value", n_iter: int=10, n_jobs: int=1, seed: Optional[int]=None, random_state: Optional[int]=None) -> None:
        """Initialize Class."""
        self.cv = cv
        self.estimator = estimator
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_dist = param_dist
        self.param_dist_pom = param_dist_pom
        self.pom = pom
        self.scoring = scoring
        self.seed = seed
        self.random_state = random_state

    def run_cv(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, mu: Optional[np.ndarray]=None, ps: Optional[np.ndarray]=None) -> OptunaSearchCV:
        """Fit."""
        # X, y = check_X_y(X, y)
        # random_state = check_random_state(self.random_state)
        # seed = random_state.randint(0, np.iinfo(np.int32).max)
        objective = Objective(self.estimator, X, y, w, mu, ps, self.param_dist, self.pom, self.param_dist_pom, self.cv, self.scoring)
        self.sampler_ = TPESampler(seed=self.seed)
        self.study_ = create_study(sampler=self.sampler_)
        # self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        # run hyper-paramter optimization.
        self.study_.optimize(objective, n_jobs=self.n_jobs, n_trials=self.n_iter)

        self.best_params_ = self.study_.best_params
        # self.best_estimator_ = self.estimator(**self.best_params_)
        # self.best_estimator_.fit(X, y)

        self.trials_dataframe = self.study_.trials_dataframe()

        return self


def bootstrap_test_score(estimator: BaseEstimator,
                         X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
                         X_test: np.ndarray,  y_test: np.ndarray, w_test: np.ndarray,
                         mu_test: Optional[np.ndarray]=None, ps_test: Optional[np.ndarray]=None,
                         n_iter: int=100) -> List[float]:
    """Compute the bootstrap metrics."""
    # ========== Fit ==========
    estimator.fit(X_train, y_train, w_train)
    # =========================

    # Compute the bootstrap scores
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
