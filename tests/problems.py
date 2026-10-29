"""Optimization problems with known solutions, which the tests share."""

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

import xipopt as xip


class KnownProblem(NamedTuple):
    """A problem with its starting point and the solution published for it."""

    name: str
    problem: xip.Problem
    start: Any
    objective: float
    x: Any


def hs071() -> KnownProblem:
    """Hock and Schittkowski problem 71, the example of the Ipopt documentation."""

    def objective(x, params):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    problem = xip.Problem(
        variables=xip.Variable(lower=1.0, upper=5.0, shape=(4,)),
        objective=objective,
        constraints=(
            xip.Constraint(lambda x, params: jnp.prod(x), lower=25.0),
            xip.Constraint(lambda x, params: jnp.sum(x**2), lower=40.0, upper=40.0),
        ),
    )
    return KnownProblem(
        "hs071",
        problem,
        np.array([1.0, 5.0, 5.0, 1.0]),
        17.0140173,
        np.array([1.0, 4.74299963, 3.82114998, 1.37940829]),
    )


def hs035() -> KnownProblem:
    """Hock and Schittkowski problem 35, a convex quadratic with one linear constraint."""

    def objective(x, params):
        return (
            9.0
            - 8.0 * x[0]
            - 6.0 * x[1]
            - 4.0 * x[2]
            + 2.0 * x[0] ** 2
            + 2.0 * x[1] ** 2
            + x[2] ** 2
            + 2.0 * x[0] * x[1]
            + 2.0 * x[0] * x[2]
        )

    problem = xip.Problem(
        variables=xip.Variable(lower=0.0, shape=(3,)),
        objective=objective,
        constraints=(
            xip.Constraint(
                lambda x, params: x[0] + x[1] + 2.0 * x[2], upper=3.0
            ),
        ),
    )
    return KnownProblem(
        "hs035",
        problem,
        np.array([0.5, 0.5, 0.5]),
        1.0 / 9.0,
        np.array([4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0]),
    )


def hs021() -> KnownProblem:
    """Hock and Schittkowski problem 21, whose solution lies on a bound and a constraint."""
    problem = xip.Problem(
        variables=xip.Variable(
            lower=np.array([2.0, -50.0]), upper=np.array([50.0, 50.0])
        ),
        objective=lambda x, params: 0.01 * x[0] ** 2 + x[1] ** 2 - 100.0,
        constraints=(
            xip.Constraint(lambda x, params: 10.0 * x[0] - x[1], lower=10.0),
        ),
    )
    return KnownProblem(
        "hs021", problem, np.array([-1.0, 1.0]), -99.96, np.array([2.0, 0.0])
    )


def hs006() -> KnownProblem:
    """Hock and Schittkowski problem 6, which has a nonlinear equality constraint."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: (1.0 - x[0]) ** 2,
        constraints=(
            xip.Constraint(
                lambda x, params: 10.0 * (x[1] - x[0] ** 2), lower=0.0, upper=0.0
            ),
        ),
    )
    return KnownProblem("hs006", problem, np.array([-1.2, 1.0]), 0.0, np.ones(2))


def hs014() -> KnownProblem:
    """Hock and Schittkowski problem 14, with an equality and an inequality constraint."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        constraints=(
            xip.Constraint(
                lambda x, params: x[0] - 2.0 * x[1], lower=-1.0, upper=-1.0
            ),
            xip.Constraint(
                lambda x, params: -(x[0] ** 2) / 4.0 - x[1] ** 2 + 1.0, lower=0.0
            ),
        ),
    )
    return KnownProblem(
        "hs014",
        problem,
        np.array([2.0, 2.0]),
        1.3934649,
        np.array([0.8228757, 0.9114378]),
    )


def all_known_problems() -> list[KnownProblem]:
    """Returns every problem with a known solution."""
    return [hs071(), hs035(), hs021(), hs006(), hs014()]


def bratu_control(size: int, coupling: float = 1.0) -> xip.Problem:
    """Returns a control problem whose state solves the Bratu equation.

    The state `u` on `size` interior points of the unit interval satisfies
    `-u'' = coupling * exp(u) + c` for the control `c`. The objective tracks a parabola.
    Both Jacobian and Hessian are banded.
    """
    spacing = 1.0 / (size + 1)
    grid = np.linspace(spacing, 1.0 - spacing, size)
    target = 0.5 * grid * (1.0 - grid)

    def objective(x, params):
        return jnp.sum((x["u"] - target) ** 2) * spacing + 1e-2 * jnp.sum(
            x["c"] ** 2
        ) * spacing

    def equation(x, params):
        u = x["u"]
        padded = jnp.concatenate([jnp.zeros(1), u, jnp.zeros(1)])
        second = (2.0 * u - padded[:-2] - padded[2:]) / spacing**2
        return second - coupling * jnp.exp(u) - x["c"]

    return xip.Problem(
        variables={"u": xip.Variable(shape=(size,)), "c": xip.Variable(shape=(size,))},
        objective=objective,
        constraints=(xip.Constraint(equation, lower=0.0, upper=0.0),),
    )


def bratu_pattern(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the coordinates of the Jacobian of `bratu_control`, in its row-major order.

    Dictionaries flatten in the order of their keys, so the controls `c` come first and the
    states `u` follow.
    """
    rows, columns = [], []
    for row in range(size):
        rows.append(row)
        columns.append(row)
        for neighbor in (row - 1, row, row + 1):
            if 0 <= neighbor < size:
                rows.append(row)
                columns.append(size + neighbor)
    return np.array(rows), np.array(columns)
