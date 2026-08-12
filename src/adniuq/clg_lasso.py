from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

Column = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _trust_region_bound(r: np.ndarray, delta: np.ndarray) -> np.ndarray:
    abs_r = np.abs(r)
    return np.where(
        abs_r <= delta,
        0.25,
        1.0 / (2.0 + np.exp(abs_r - delta) + np.exp(delta - abs_r)),
    )


@dataclass
class CLGLasso:
    lam: float = 1.0
    max_iter: int = 100
    tol: float = 5e-4
    fit_intercept: bool = True

    beta_: np.ndarray = field(init=False, default=None)
    intercept_: float = field(init=False, default=0.0)
    n_iter_: int = field(init=False, default=0)
    converged_: bool = field(init=False, default=False)

    def fit(self, columns: List[Column], y: np.ndarray, n_samples: int) -> "CLGLasso":
        n_features = len(columns)
        beta = np.zeros(n_features)
        delta = np.ones(n_features)
        r = np.zeros(n_samples)
        intercept = 0.0
        delta_b = 1.0

        for iteration in range(1, self.max_iter + 1):
            sum_abs_delta_r = 0.0

            if self.fit_intercept:
                d_b = self._intercept_step(r, y, delta_b)
                if d_b != 0.0:
                    r += d_b * y
                    sum_abs_delta_r += np.sum(np.abs(d_b * y))
                    intercept += d_b
                    delta_b = max(2.0 * abs(d_b), delta_b / 2.0)

            for j in range(n_features):
                rows, vals, vals_sq = columns[j]
                if rows.size == 0:
                    continue

                d_beta = self._coordinate_step(
                    r[rows], vals, vals_sq, y[rows], beta[j], self.lam, delta[j]
                )
                if d_beta == 0.0:
                    continue

                delta_r = d_beta * vals * y[rows]
                r[rows] += delta_r
                sum_abs_delta_r += np.sum(np.abs(delta_r))
                beta[j] += d_beta
                delta[j] = max(2.0 * abs(d_beta), delta[j] / 2.0)

            if sum_abs_delta_r / (1.0 + np.sum(np.abs(r))) <= self.tol:
                self.converged_ = True
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.beta_ = beta
        self.intercept_ = intercept
        return self

    @staticmethod
    def _coordinate_step(r_sub, vals, vals_sq, y_sub, beta_j, lam, delta_j) -> float:
        def tentative(sign: float) -> float:
            correction = 1.0 / (1.0 + np.exp(np.clip(r_sub, -500, 500)))
            grad = float(np.dot(vals * y_sub, correction)) - lam * sign
            bound = _trust_region_bound(r_sub, delta_j * np.abs(vals))
            denom = float(np.dot(vals_sq, bound))
            return 0.0 if denom < 1e-14 else grad / denom

        if beta_j == 0.0:
            step = tentative(+1.0)
            if step <= 0.0:
                step = tentative(-1.0)
                if step >= 0.0:
                    step = 0.0
        else:
            step = tentative(float(np.sign(beta_j)))
            if (beta_j + step) * beta_j < 0.0:
                step = -beta_j

        return float(np.clip(step, -delta_j, delta_j))

    @staticmethod
    def _intercept_step(r: np.ndarray, y: np.ndarray, delta_b: float) -> float:
        correction = 1.0 / (1.0 + np.exp(np.clip(r, -500, 500)))
        grad = float(np.dot(y, correction))
        denom = float(np.sum(_trust_region_bound(r, np.full_like(r, delta_b))))
        if denom < 1e-14:
            return 0.0
        return float(np.clip(grad / denom, -delta_b, delta_b))

    def decision_function(self, columns: List[Column], n_samples: int) -> np.ndarray:
        scores = np.full(n_samples, self.intercept_)
        for j in np.flatnonzero(self.beta_):
            rows, vals, _ = columns[j]
            scores[rows] += self.beta_[j] * vals
        return scores

    def predict_proba(self, columns: List[Column], n_samples: int) -> np.ndarray:
        scores = np.clip(self.decision_function(columns, n_samples), -500, 500)
        return 1.0 / (1.0 + np.exp(-scores))

    @property
    def n_nonzero(self) -> int:
        return int(np.count_nonzero(self.beta_))
