"""Tests that Hock and Schittkowski problems reach their published solutions."""

import numpy as np
import pytest
from problems import REQUIRES_ASDEX, KnownProblem, all_known_problems

import xipopt as xip

PROBLEMS = {known.name: known for known in all_known_problems()}


@pytest.mark.parametrize(
    "derivatives",
    [
        pytest.param(derivatives, marks=REQUIRES_ASDEX)
        if derivatives is xip.Derivatives.SPARSE
        else derivatives
        for derivatives in xip.Derivatives
    ],
)
@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_solution_matches_publication(name: str, derivatives: xip.Derivatives) -> None:
    """Checks the objective and the point for every way of computing derivatives."""
    known: KnownProblem = PROBLEMS[name]
    solution = xip.solve(
        known.problem,
        known.start,
        compile_options=xip.CompileOptions(derivatives=derivatives),
    )
    assert solution.status is xip.Status.SOLVE_SUCCEEDED
    assert solution.status.succeeded
    assert solution.objective == pytest.approx(known.objective, abs=1e-6)
    np.testing.assert_allclose(solution.x, known.x, atol=1e-5)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_limited_memory_reaches_the_same_objective(name: str) -> None:
    """Checks that approximating the Hessian still solves, though in more iterations."""
    known = PROBLEMS[name]
    solution = xip.solve(
        known.problem,
        known.start,
        options=xip.SolveOptions(tolerance=1e-9),
        compile_options=xip.CompileOptions(hessian=xip.Hessian.LIMITED_MEMORY),
    )
    assert solution.status.succeeded
    assert solution.objective == pytest.approx(known.objective, abs=1e-4)
    assert solution.statistics.hessian.calls == 0


def test_solution_describes_the_solve() -> None:
    """Checks the constraints, multipliers and counters of a solution of HS071."""
    known = PROBLEMS["hs071"]
    solution = xip.solve(xip.compile(known.problem), known.start)
    product, squares = solution.constraints
    assert product >= 25.0 - 1e-6
    assert squares == pytest.approx(40.0)
    assert len(solution.constraint_multipliers) == 2
    # The first variable sits on its lower bound, which is the only active bound.
    assert solution.lower_multipliers[0] > 1e-3
    assert np.all(solution.upper_multipliers < 1e-6)
    assert solution.iterations == 8
    assert solution.statistics.objective.calls >= solution.iterations
    assert solution.statistics.hessian.calls == solution.iterations
    assert solution.solving_seconds > 0.0
    assert solution.build == xip.BuildStatistics()
