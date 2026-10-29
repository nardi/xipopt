"""Tests that random convex problems agree with SciPy."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize

import xipopt as xip


def random_instance(seed: int, size: int, rows: int) -> dict[str, np.ndarray]:
    """Returns the data of a convex problem with linear and ball constraints."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(size, size))
    return {
        "hessian": factor @ factor.T / size + np.eye(size),
        "cost": rng.normal(size=size),
        "matrix": rng.normal(size=(rows, size)),
        "limit": rng.uniform(1.0, 2.0, size=rows),
    }


def objective(x: Any, params: dict[str, Any]) -> Any:
    """Returns a quadratic plus a smooth convex term."""
    return (
        0.5 * x @ params["hessian"] @ x
        + params["cost"] @ x
        + jnp.sum(jnp.log1p(jnp.exp(x)))
    )


def reference_solution(data: dict[str, np.ndarray], radius: float) -> float:
    """Solves the problem with SciPy's SLSQP and returns the objective."""

    def value(x: np.ndarray) -> float:
        return float(
            0.5 * x @ data["hessian"] @ x
            + data["cost"] @ x
            + np.sum(np.log1p(np.exp(x)))
        )

    size = data["cost"].size
    result = optimize.minimize(
        value,
        np.zeros(size),
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": lambda x: data["limit"] - data["matrix"] @ x},
            {"type": "ineq", "fun": lambda x: radius**2 - x @ x},
        ],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert result.success
    return float(result.fun)


@pytest.mark.parametrize("seed", range(5))
def test_objective_matches_scipy(seed: int) -> None:
    """Checks the optimal value of random problems, with the data as parameters."""
    size, rows, radius = 6, 4, 1.5
    data = random_instance(seed, size, rows)
    problem = xip.Problem(
        variables=xip.Variable(shape=(size,)),
        objective=objective,
        constraints=(
            xip.Constraint(lambda x, params: params["matrix"] @ x, upper=data["limit"]),
            xip.Constraint(lambda x, params: jnp.sum(x**2), upper=radius**2),
        ),
    )
    solution = xip.solve(
        problem, params=data, options=xip.SolveOptions(tolerance=1e-10)
    )
    assert solution.status.succeeded
    assert solution.objective == pytest.approx(
        reference_solution(data, radius), rel=1e-6
    )


def test_compiled_problem_solves_other_parameters() -> None:
    """Checks that one compilation serves any parameter values of the same layout."""
    size, rows, radius = 5, 3, 1.5
    problem = xip.Problem(
        variables=xip.Variable(shape=(size,)),
        objective=objective,
        constraints=(
            xip.Constraint(lambda x, params: params["matrix"] @ x, upper=1.0),
            xip.Constraint(lambda x, params: jnp.sum(x**2), upper=radius**2),
        ),
    )
    first = random_instance(0, size, rows)
    compiled = xip.compile(problem, first)
    assert compiled.statistics.compiled == 5
    for seed in (1, 2):
        data = random_instance(seed, size, rows)
        solution = xip.solve(compiled, params=data)
        assert solution.status.succeeded
        # The constraint holds with the matrix of the new parameters.
        assert np.all(data["matrix"] @ solution.x <= 1.0 + 1e-6)
