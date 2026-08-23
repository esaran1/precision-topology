"""Minimal CMA-ES, implemented in numpy for the direct weight search.

Standard (mu/mu_w, lambda) formulation after Hansen's tutorial: weighted
recombination, cumulative step-size adaptation, rank-one and rank-mu
covariance updates.  No external dependencies; deterministic given a seed.

This exists because scipy is not in the environment and the search must not
be SGD.  It is unit-tested on quadratic and Rosenbrock objectives before it
is allowed near the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


@dataclass
class CMAResult:
    best_x: np.ndarray
    best_f: float
    evaluations: int
    generations: int
    converged: bool


def cma_es(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    sigma0: float,
    max_generations: int = 500,
    population: int | None = None,
    seed: int = 0,
    target_f: float | None = None,
    sigma_floor: float = 1e-12,
) -> CMAResult:
    """Minimize ``objective`` starting from ``x0`` with initial step ``sigma0``.

    Stops early when ``target_f`` is reached (useful when the objective is an
    error count and 0 is the goal) or when the step size collapses.
    """

    n = x0.size
    rng = np.random.default_rng(seed)
    lam = population if population is not None else 4 + int(3 * math.log(n))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / weights.sum()
    mu_eff = 1.0 / float((weights**2).sum())

    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1.0 - c_1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
    )
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    mean = x0.astype(np.float64).copy()
    sigma = float(sigma0)
    covariance = np.eye(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    best_x = mean.copy()
    best_f = float(objective(mean))
    evaluations = 1

    for generation in range(1, max_generations + 1):
        # Eigendecomposition every generation: n is small (<100) here.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        sqrt_cov = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        inv_sqrt_cov = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        z = rng.standard_normal((lam, n))
        y = z @ sqrt_cov.T
        candidates = mean[None, :] + sigma * y
        fitness = np.array([objective(candidate) for candidate in candidates])
        evaluations += lam
        order = np.argsort(fitness, kind="stable")

        if fitness[order[0]] < best_f:
            best_f = float(fitness[order[0]])
            best_x = candidates[order[0]].copy()
            if target_f is not None and best_f <= target_f:
                return CMAResult(best_x, best_f, evaluations, generation, True)

        selected_y = y[order[:mu]]
        y_w = weights @ selected_y
        mean = mean + sigma * y_w

        p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(
            c_sigma * (2.0 - c_sigma) * mu_eff
        ) * (inv_sqrt_cov @ y_w)
        sigma_norm = float(np.linalg.norm(p_sigma))
        h_sigma = float(
            sigma_norm
            / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (generation + 1)))
            < (1.4 + 2.0 / (n + 1.0)) * chi_n
        )
        p_c = (1.0 - c_c) * p_c + h_sigma * math.sqrt(
            c_c * (2.0 - c_c) * mu_eff
        ) * y_w

        rank_mu = (selected_y * weights[:, None]).T @ selected_y
        covariance = (
            (1.0 - c_1 - c_mu) * covariance
            + c_1
            * (
                np.outer(p_c, p_c)
                + (1.0 - h_sigma) * c_c * (2.0 - c_c) * covariance
            )
            + c_mu * rank_mu
        )
        covariance = 0.5 * (covariance + covariance.T)

        sigma = sigma * math.exp((c_sigma / d_sigma) * (sigma_norm / chi_n - 1.0))
        if sigma < sigma_floor:
            break

    return CMAResult(best_x, best_f, evaluations, generation, False)
