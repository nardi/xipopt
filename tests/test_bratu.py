"""Tests of a banded problem, whose derivatives are sparse."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from problems import bratu_control, bratu_pattern

import xipopt as xip

SIZE = 12


def solve_with(problem: xip.Problem, derivatives: xip.Derivatives) -> xip.Solution:
    """Solves the Bratu problem with a way of computing derivatives."""
    return xip.solve(
        problem, compile_options=xip.CompileOptions(derivatives=derivatives)
    )


def residual(solution: xip.Solution) -> float:
    """Returns the largest violation of the equation by a solution."""
    return float(np.max(np.abs(solution.constraints[0])))


def test_detected_patterns_are_banded() -> None:
    """Checks that detection finds the three diagonals instead of a dense Jacobian."""
    compiled = xip.compile(bratu_control(SIZE))
    # Three entries per row of the state and one for the control, in each of the rows.
    assert compiled.structure.jacobian_rows.size == 4 * SIZE - 2
    # The Hessian is diagonal.
    assert compiled.structure.hessian_rows.size == 2 * SIZE
    assert np.all(compiled.structure.hessian_rows >= compiled.structure.hessian_columns)


GIVEN_FORMS = {
    "coordinates": lambda rows, columns: (rows, columns),
    "matrix": lambda rows, columns: sparse_matrix(rows, columns),
    "bcoo": lambda rows, columns: sparse.BCOO(
        (jnp.ones(rows.size), jnp.stack([rows, columns], axis=1)),
        shape=(SIZE, 2 * SIZE),
    ),
}
"""Ways to write the Jacobian pattern of the Bratu problem."""


def sparse_matrix(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Returns the pattern as a boolean matrix."""
    matrix = np.zeros((SIZE, 2 * SIZE), dtype=bool)
    matrix[rows, columns] = True
    return matrix


@pytest.mark.parametrize("form", sorted(GIVEN_FORMS))
@pytest.mark.parametrize("derivatives", [xip.Derivatives.SPARSE, xip.Derivatives.DENSE])
def test_given_pattern_matches_detected(form: str, derivatives: xip.Derivatives) -> None:
    """Checks that a pattern in any form gives the objective of the detected one."""
    reference = solve_with(bratu_control(SIZE), xip.Derivatives.SPARSE)
    rows, columns = bratu_pattern(SIZE)
    problem = bratu_control(SIZE)._replace(
        jacobian_pattern=GIVEN_FORMS[form](rows, columns),
        hessian_pattern=(np.arange(2 * SIZE), np.arange(2 * SIZE)),
    )
    solution = solve_with(problem, derivatives)
    assert solution.status.succeeded
    assert solution.objective == pytest.approx(reference.objective, rel=1e-8)
    assert residual(solution) < 1e-6


def test_pattern_of_asdex_is_accepted() -> None:
    """Checks that an `asdex.SparsityPattern` works as a pattern."""
    asdex = pytest.importorskip("asdex")
    rows, columns = bratu_pattern(SIZE)
    avals = (
        jax.ShapeDtypeStruct((2 * SIZE,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
    )
    pattern = asdex.SparsityPattern.from_coo(
        rows, columns, (SIZE, 2 * SIZE), input_avals=avals
    )
    reference = solve_with(bratu_control(SIZE), xip.Derivatives.SPARSE)
    solution = solve_with(
        bratu_control(SIZE)._replace(jacobian_pattern=pattern), xip.Derivatives.SPARSE
    )
    assert solution.objective == pytest.approx(reference.objective, rel=1e-8)


def test_hessian_pattern_may_list_the_upper_triangle() -> None:
    """Checks that a Hessian pattern with only upper entries is mirrored."""
    upper = (np.array([0, 1, 5]), np.array([1, 2, 7]))
    problem = bratu_control(SIZE)._replace(
        hessian_pattern=(
            np.concatenate([np.arange(2 * SIZE), upper[0]]),
            np.concatenate([np.arange(2 * SIZE), upper[1]]),
        )
    )
    compiled = xip.compile(problem, options=xip.CompileOptions())
    assert compiled.structure.hessian_rows.size == 2 * SIZE + 3
    assert np.all(compiled.structure.hessian_rows >= compiled.structure.hessian_columns)
    assert xip.solve(compiled).status.succeeded


def test_bounds_on_a_larger_problem() -> None:
    """Checks that a problem with a few hundred variables solves with sparse derivatives."""
    solution = xip.solve(bratu_control(150))
    assert solution.status.succeeded
    assert residual(solution) < 1e-6
