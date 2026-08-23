"""CMA-ES must solve standard benchmarks before it touches the experiment."""

from __future__ import annotations

import numpy as np

from src.cmaes import cma_es


def test_sphere():
    result = cma_es(
        lambda x: float((x**2).sum()),
        x0=np.full(10, 3.0),
        sigma0=1.0,
        max_generations=400,
        seed=0,
        target_f=1e-10,
    )
    assert result.best_f < 1e-10


def test_shifted_ellipsoid():
    shift = np.arange(8, dtype=np.float64)
    scales = 10.0 ** np.linspace(0, 3, 8)

    def objective(x: np.ndarray) -> float:
        return float((scales * (x - shift) ** 2).sum())

    result = cma_es(
        objective, x0=np.zeros(8), sigma0=2.0, max_generations=800, seed=1,
        target_f=1e-8,
    )
    assert result.best_f < 1e-8
    assert np.allclose(result.best_x, shift, atol=1e-3)


def test_rosenbrock():
    def rosenbrock(x: np.ndarray) -> float:
        return float(
            (100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2).sum()
        )

    result = cma_es(
        rosenbrock, x0=np.zeros(6), sigma0=0.5, max_generations=2_000, seed=2,
        target_f=1e-8,
    )
    assert result.best_f < 1e-8


def test_deterministic_given_seed():
    objective = lambda x: float((x**2).sum())  # noqa: E731
    first = cma_es(objective, np.full(5, 2.0), 1.0, max_generations=50, seed=7)
    second = cma_es(objective, np.full(5, 2.0), 1.0, max_generations=50, seed=7)
    assert np.array_equal(first.best_x, second.best_x)
    assert first.best_f == second.best_f


def test_discrete_plateau_objective():
    """CMA-ES must make progress on a count-like piecewise-constant objective
    with a smooth tie-breaker, the exact shape of the error-count search."""

    target = np.array([1.0, -2.0, 0.5, 3.0])

    def objective(x: np.ndarray) -> float:
        distance = float(np.abs(x - target).max())
        count = float(np.floor(distance * 10.0))  # steps of 0.1
        return count + distance / (1.0 + distance)

    result = cma_es(
        objective, x0=np.zeros(4), sigma0=1.0, max_generations=400, seed=3,
        target_f=0.05,
    )
    assert result.best_f < 0.05
