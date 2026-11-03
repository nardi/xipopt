"""Tests of problems whose functions call back into Python."""

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

import xipopt as xip

OPTIONS = xip.CompileOptions(
    host_callbacks=True, derivatives=xip.Derivatives.DENSE, disk_cache=False
)
"""Compile options for callbacks, with dense derivatives since `asdex` does not know them."""


def problem_with_check(check: Any) -> xip.Problem:
    """Returns a problem that minimizes `(x - 1) ** 2` and calls `check` with every point."""

    def objective(x: Any, params: Any) -> Any:
        jax.debug.callback(check, x)
        return (x[0] - 1.0) ** 2

    return xip.Problem(variables=xip.Variable(shape=(1,)), objective=objective)


def test_callbacks_see_the_points_of_a_solve() -> None:
    """Checks that a problem with a callback solves and that the callback sees the iterates."""
    points: list[float] = []
    compiled = xip.compile(
        problem_with_check(lambda x: points.append(float(x[0]))), options=OPTIONS
    )
    solution = xip.solve(compiled, np.array([-4.0]))
    assert solution.status is xip.Status.SOLVE_SUCCEEDED
    assert solution.x[0] == pytest.approx(1.0, abs=1e-6)
    assert points[0] == -4.0
    assert len(points) >= solution.statistics.objective.calls


def test_a_raising_callback_makes_ipopt_cut_the_step() -> None:
    """Checks that an error in a callback at a trial point rejects the point, as `error_if` does."""
    raised: list[float] = []

    def check(x: Any) -> None:
        # The first point beyond 0.9 is rejected once, like a failed check of a library.
        if not raised and float(x[0]) > 0.9:
            raised.append(float(x[0]))
            raise ValueError("the check failed")

    compiled = xip.compile(problem_with_check(check), options=OPTIONS)
    solution = xip.solve(compiled, np.array([0.0]))
    assert len(raised) == 1
    assert solution.status is xip.Status.SOLVE_SUCCEEDED
    assert solution.x[0] == pytest.approx(1.0, abs=1e-6)


def test_a_callback_that_always_raises_ends_the_solve_with_its_message() -> None:
    """Checks that the message of the callback reaches the caller when the solve fails."""

    def check(x: Any) -> None:
        raise ValueError("always wrong")

    compiled = xip.compile(problem_with_check(check), options=OPTIONS)
    with pytest.raises(RuntimeError, match="ValueError: always wrong"):
        xip.solve(compiled, np.array([0.0]))


def test_problem_with_callbacks_cannot_be_saved(tmp_path: Path) -> None:
    """Checks that saving explains why a problem with callbacks has no file."""
    compiled = xip.compile(problem_with_check(lambda x: None), options=OPTIONS)
    with pytest.raises(ValueError, match=r"host_callbacks=True.*cannot be saved"):
        xip.save(compiled, tmp_path / "problem.xip")
    with pytest.raises(ValueError, match="cannot be saved"):
        xip.dumps(compiled)


def test_callbacks_need_the_option() -> None:
    """Checks that the error for a callback names the way to allow it."""
    options = OPTIONS._replace(host_callbacks=False)
    with pytest.raises(Exception, match="host_callbacks=True"):
        xip.compile(problem_with_check(lambda x: None), options=options)
