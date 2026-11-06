"""Tests of the joint evaluation: one program per point for the values and the derivatives."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import SolveBounds, hs071

import xipopt as xip

INFEASIBLE_BOUNDS = SolveBounds(
    variable_bounds=xip.Bounds(-5.0, 5.0), constraint_bounds=xip.Bounds(4.0, 4.0)
)
"""The bounds of the problems from `infeasible_where`, as keyword arguments of `xip.solve`."""


def test_callbacks_at_one_point_share_one_execution() -> None:
    """Checks that the four first-order callbacks cost one execution per distinct point."""
    known = hs071()
    solution = xip.solve(known.problem, known.start, **known.bounds)
    statistics = solution.statistics
    callbacks = (
        statistics.objective,
        statistics.gradient,
        statistics.constraints,
        statistics.jacobian,
    )
    misses = sum(item.calls - item.cache_hits for item in callbacks)
    assert misses == statistics.first_order_program.executions
    assert sum(item.cache_hits for item in callbacks) > 0
    assert statistics.first_order_program.executions < sum(
        item.calls for item in callbacks
    )
    assert statistics.hessian_program.executions == statistics.hessian.calls
    assert statistics.hessian.cache_hits == 0


def test_a_new_solve_does_not_reuse_the_values_of_the_last() -> None:
    """Checks that parameters that change between solves from one point are not mixed up."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(1,)),
        evaluate=xip.evaluation_from_functions(
            objective=lambda x, p: jnp.sum((x - p) ** 2)
        ),
    )
    compiled = xip.compile(problem, np.zeros(1))
    for target in (3.0, -2.0, 3.0):
        solution = xip.solve(compiled, np.array([0.0]), params=np.array([target]))
        np.testing.assert_allclose(solution.x, [target], atol=1e-6)


def test_derivatives_that_are_not_numbers_reject_the_point() -> None:
    """Checks that a NaN gradient at a trial point cuts the step, as a NaN value would."""

    @jax.custom_jvp
    def objective(x: jax.Array) -> jax.Array:
        return (x[0] - 3.0) ** 2

    @objective.defjvp
    def objective_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
        (x,), (tangent,) = primals, tangents
        slope = jnp.where(x[0] > 1.5, jnp.nan, 2.0 * (x[0] - 3.0))
        return objective(x), slope * tangent[0]

    problem = xip.Problem(
        variables=xip.Variable(shape=(1,)),
        evaluate=lambda x, p: xip.Evaluation(objective(x)),
    )
    options = xip.CompileOptions(
        derivatives=xip.Derivatives.DENSE, hessian=xip.Hessian.LIMITED_MEMORY
    )
    solution = xip.solve(
        problem,
        np.array([0.0]),
        options=xip.SolveOptions(max_iterations=30),
        compile_options=options,
    )
    assert solution.status is not xip.Status.INVALID_NUMBER_DETECTED
    assert solution.x[0] <= 1.5 + 1e-6
    assert solution.statistics.gradient.calls > 0


def test_a_shared_computation_runs_once_per_execution() -> None:
    """Checks that one `evaluate` that shares work needs less than separate functions."""
    counts = {"shared": 0, "separate": 0}

    def count(name: str) -> Any:
        def increment(value: Any) -> None:
            counts[name] += 1

        return increment

    def inner(x: Any, name: str) -> Any:
        jax.debug.callback(count(name), x)
        return jnp.sin(x)

    def shared(x: Any, params: Any) -> xip.Evaluation:
        state = inner(x, "shared")
        return xip.Evaluation(jnp.sum(state**2), (jnp.sum(state)[None],))

    separate = xip.evaluation_from_functions(
        objective=lambda x, params: jnp.sum(inner(x, "separate") ** 2),
        constraints=(lambda x, params: jnp.sum(inner(x, "separate"))[None],),
    )
    options = xip.CompileOptions(
        host_callbacks=True,
        derivatives=xip.Derivatives.DENSE,
        hessian=xip.Hessian.LIMITED_MEMORY,
        disk_cache=False,
    )
    executions = {}
    for name, evaluate in (("shared", shared), ("separate", separate)):
        problem = xip.Problem(variables=xip.Variable(shape=(2,)), evaluate=evaluate)
        solution = xip.solve(
            problem,
            np.array([0.5, 0.5]),
            compile_options=options,
            variable_bounds=xip.Bounds(-1.0, 1.0),
            constraint_bounds=xip.Bounds(lower=0.1),
        )
        assert solution.status.succeeded
        executions[name] = solution.statistics.first_order_program.executions
    assert counts["shared"] == executions["shared"]
    assert counts["separate"] == 2 * executions["separate"]


def test_evaluation_must_match_the_bounds() -> None:
    """Checks the errors for an evaluation that does not match the problem or the bounds."""
    variables = xip.Variable(shape=(2,))
    with pytest.raises(TypeError, match="must return an Evaluation"):
        xip.compile(xip.Problem(variables, evaluate=lambda x, p: jnp.zeros(())))  # ty: ignore[invalid-argument-type]
    problem = xip.Problem(
        variables, evaluate=lambda x, p: xip.Evaluation(jnp.zeros(()), (x, x))
    )
    with pytest.raises(ValueError, match="`constraint_bounds` is required"):
        xip.solve(problem, np.zeros(2))
    with pytest.raises(ValueError, match="has structure"):
        xip.solve(problem, np.zeros(2), constraint_bounds=xip.Bounds(lower=(0.0,)))


def infeasible_where(region: Callable[[jax.Array], jax.Array]) -> xip.Problem:
    """Returns a problem whose equality constraint is not a number inside a region.

    The constraint can only hold at a point with `x[0] > 1`, so regions that cover those
    points make Ipopt search for a feasible point among points where evaluation fails.
    """

    def evaluate(x: jax.Array, params: None) -> xip.Evaluation:
        constraint = jnp.where(region(x), jnp.nan, x[0] ** 2 + x[1] ** 2)
        return xip.Evaluation((x[0] - 3.0) ** 2 + x[1] ** 2, (constraint,))

    return xip.Problem(xip.Variable(shape=(2,)), evaluate)


def test_constraints_at_failed_points_are_rejected() -> None:
    """Checks that Ipopt can ask for the constraints alone at a point where they fail.

    The restoration phase evaluates the constraints without the objective, so the callback
    for them is the first to meet a failed point. It has to report the failure.
    """
    solution = xip.solve(
        infeasible_where(lambda x: x[0] > 1.0),
        x0=np.array([0.5, 0.5]),
        options=xip.SolveOptions(max_iterations=60),
        **INFEASIBLE_BOUNDS,
    )
    statistics = solution.statistics
    assert statistics.constraints.calls > statistics.objective.calls
    assert not solution.status.succeeded


def test_failure_at_the_starting_point_ends_the_solve() -> None:
    """Checks that the derivatives at a failed starting point end the solve cleanly."""
    solution = xip.solve(
        infeasible_where(lambda x: jnp.all(x == 0.0)), **INFEASIBLE_BOUNDS
    )
    assert solution.status is xip.Status.INVALID_NUMBER_DETECTED
    assert solution.statistics.jacobian.calls > 0


def test_older_asdex_without_values_solves_the_same_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the fallback for a version of asdex that cannot return values with a Jacobian."""
    import asdex

    known = hs071()
    options = xip.CompileOptions(derivatives=xip.Derivatives.SPARSE, disk_cache=False)
    expected = xip.solve(known.problem, compile_options=options, **known.bounds)
    monkeypatch.delattr(asdex, "value_and_jacobian_from_coloring")
    solution = xip.solve(known.problem, compile_options=options, **known.bounds)
    assert solution.status.succeeded
    np.testing.assert_allclose(solution.x, expected.x, atol=1e-6)
