"""Tests of problems that carry a state from one evaluation to the next."""

import jax.numpy as jnp
import numpy as np
import pytest

import xipopt as xip

TOLERANCE = 1e-7
"""How closely the solutions of the stateful and the stateless forms must agree."""

NORM_BOUNDS = xip.Bounds(upper=2.0)
"""The constraint bounds of the norm problems, which bound the inner root from above."""

COUNTING_BOUNDS = xip.Bounds(0.0, 3.0)
"""The variable bounds of `counting_problem`."""


def newton_square_root(
    target: jnp.ndarray, start: jnp.ndarray, steps: int
) -> jnp.ndarray:
    """Returns the square root of `target` from Newton steps that begin at `start`."""
    root = start
    for _ in range(steps):
        root = 0.5 * (root + target / root)
    return root


def distance_objective(x: jnp.ndarray) -> jnp.ndarray:
    """Returns the squared distance of `x` to the point (2, 1)."""
    return (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2


def stateless_norm_problem() -> xip.Problem:
    """Returns a problem whose constraint is an inner Newton solve from a fixed start."""

    def evaluate(x: jnp.ndarray, params: None) -> xip.Evaluation:
        root = newton_square_root(jnp.sum(x**2) + 1.0, jnp.asarray(1.0), 40)
        return xip.Evaluation(distance_objective(x), (root,))

    return xip.Problem(xip.Variable(shape=(2,)), evaluate)


def stateful_norm_problem() -> xip.StatefulProblem:
    """Returns the same problem, with the inner solve starting from the previous root."""

    def evaluate(
        x: jnp.ndarray, params: None, state: jnp.ndarray
    ) -> tuple[xip.Evaluation, jnp.ndarray]:
        root = newton_square_root(jnp.sum(x**2) + 1.0, state, 40)
        return xip.Evaluation(distance_objective(x), (root,)), root

    return xip.StatefulProblem(xip.Variable(shape=(2,)), evaluate, jnp.asarray(1.0))


def counting_problem(
    failing_above: float = np.inf,
) -> xip.StatefulProblem:
    """Returns a problem whose state counts the successful evaluations of the program.

    The objective is not a number above `failing_above`, which makes those evaluations fail.
    """

    def evaluate(
        x: jnp.ndarray, params: None, state: dict[str, jnp.ndarray]
    ) -> tuple[xip.Evaluation, dict[str, jnp.ndarray]]:
        objective = jnp.where(x[0] > failing_above, jnp.nan, -x[0] + x[1] ** 2)
        return xip.Evaluation(objective), {"count": state["count"] + 1.0}

    return xip.StatefulProblem(
        xip.Variable(shape=(2,)), evaluate, {"count": jnp.zeros(())}
    )


def test_state_does_not_change_the_solution() -> None:
    """Checks that a state that only chooses a starting guess leaves the solution as it is."""
    plain = xip.solve(stateless_norm_problem(), constraint_bounds=NORM_BOUNDS)
    stateful = xip.solve(stateful_norm_problem(), constraint_bounds=NORM_BOUNDS)
    assert plain.status.succeeded and stateful.status.succeeded
    np.testing.assert_allclose(stateful.x, plain.x, atol=TOLERANCE)
    np.testing.assert_allclose(stateful.objective, plain.objective, atol=TOLERANCE)
    # The state ends at the inner root of the last point that was evaluated.
    assert stateful.state is not None
    np.testing.assert_allclose(
        stateful.state, np.sqrt(np.sum(np.asarray(stateful.x) ** 2) + 1.0), atol=1e-5
    )


def test_derivatives_ignore_the_state() -> None:
    """Checks that the Jacobian is the same whatever the state holds."""
    plain = xip.compile(stateless_norm_problem())
    stateful = xip.compile(stateful_norm_problem())
    assert stateful.structure.jacobian_rows.tolist() == (
        plain.structure.jacobian_rows.tolist()
    )
    assert stateful.state is not None
    assert plain.state is None


@pytest.mark.parametrize(
    "derivatives",
    [xip.Derivatives.DENSE, xip.Derivatives.SPARSE],
    ids=["dense", "sparse"],
)
def test_state_is_the_count_of_evaluations(derivatives: xip.Derivatives) -> None:
    """Checks that each execution of the first-order program updates the state once."""
    solution = xip.solve(
        counting_problem(),
        compile_options=xip.CompileOptions(derivatives=derivatives),
        variable_bounds=COUNTING_BOUNDS,
    )
    assert solution.status.succeeded
    executions = solution.statistics.first_order_program.executions
    assert solution.state is not None
    assert solution.state["count"] == executions
    # The Hessian program ran too and did not change the state.
    assert solution.statistics.hessian_program.executions > 0


def test_a_solve_starts_from_the_initial_state() -> None:
    """Checks that the state is reset at the start of every solve."""
    compiled = xip.compile(counting_problem())
    first = xip.solve(compiled, variable_bounds=COUNTING_BOUNDS)
    second = xip.solve(compiled, variable_bounds=COUNTING_BOUNDS)
    assert first.state is not None and second.state is not None
    assert first.state["count"] == first.statistics.first_order_program.executions
    assert second.state["count"] == second.statistics.first_order_program.executions


def test_a_warm_start_continues_from_the_state() -> None:
    """Checks that the state of the solution that is warm started from is the new start."""
    compiled = xip.compile(counting_problem())
    first = xip.solve(compiled, variable_bounds=COUNTING_BOUNDS)
    second = xip.solve(
        compiled, x0=first.x, warm_start=first, variable_bounds=COUNTING_BOUNDS
    )
    assert first.state is not None and second.state is not None
    executions = second.statistics.first_order_program.executions
    assert second.state["count"] == first.state["count"] + executions


def test_failed_evaluations_keep_the_state() -> None:
    """Checks that a point where the program fails does not update the state."""
    # The optimum is at the edge of the region where evaluations fail, so the solve may end
    # without converging. The state is what the test is about.
    solution = xip.solve(
        counting_problem(failing_above=1.5),
        options=xip.SolveOptions(max_iterations=30),
        variable_bounds=COUNTING_BOUNDS,
    )
    assert solution.state is not None
    assert np.isfinite(solution.state["count"])
    assert solution.state["count"] < solution.statistics.first_order_program.executions


def test_state_survives_saving_and_loading() -> None:
    """Checks that a loaded problem starts from the stored initial state."""
    compiled = xip.compile(counting_problem())
    variables = counting_problem().variables
    loaded = xip.loads(
        xip.dumps(compiled), variables, None, state={"count": np.zeros(())}
    )
    solution = xip.solve(loaded, variable_bounds=COUNTING_BOUNDS)
    expected = xip.solve(compiled, variable_bounds=COUNTING_BOUNDS)
    assert solution.state is not None and expected.state is not None
    assert solution.state["count"] == expected.state["count"]
    np.testing.assert_allclose(solution.x, expected.x)


def test_loading_checks_the_state() -> None:
    """Checks the errors for a missing, an extra and a mismatched state."""
    stateful = xip.dumps(xip.compile(counting_problem()))
    stateless = xip.dumps(xip.compile(stateless_norm_problem()))
    with pytest.raises(ValueError, match="has state, so `state` is required"):
        xip.loads(stateful, counting_problem().variables)
    with pytest.raises(ValueError, match="no state, so `state` must be left out"):
        xip.loads(
            stateless,
            xip.Variable(shape=(2,)),
            constraints=(np.zeros(()),),
            state=jnp.zeros(()),
        )
    with pytest.raises(ValueError, match="state"):
        xip.loads(
            stateful, counting_problem().variables, None, state={"other": np.zeros(())}
        )


def test_state_must_keep_its_structure() -> None:
    """Checks the error for an evaluation that returns a state of another shape."""

    def evaluate(
        x: jnp.ndarray, params: None, state: jnp.ndarray
    ) -> tuple[xip.Evaluation, jnp.ndarray]:
        return xip.Evaluation(jnp.sum(x**2)), jnp.zeros(2)

    problem = xip.StatefulProblem(xip.Variable(shape=(2,)), evaluate, jnp.zeros(()))
    with pytest.raises(ValueError, match="structure and shapes"):
        xip.compile(problem)


def test_state_must_be_nonempty_and_floating_point() -> None:
    """Checks the errors for an empty state and an integer state."""

    def evaluate(
        x: jnp.ndarray, params: None, state: object
    ) -> tuple[xip.Evaluation, object]:
        return xip.Evaluation(jnp.sum(x**2)), state

    variables = xip.Variable(shape=(2,))
    with pytest.raises(ValueError, match="at least one element"):
        xip.compile(xip.StatefulProblem(variables, evaluate, {}))
    with pytest.raises(TypeError, match="floating point"):
        xip.compile(xip.StatefulProblem(variables, evaluate, jnp.zeros((), jnp.int32)))


def test_older_asdex_without_values_still_carries_the_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the fallback for a version of asdex that cannot return values with a Jacobian."""
    import asdex

    monkeypatch.delattr(asdex, "value_and_jacobian_from_coloring")
    solution = xip.solve(
        stateful_norm_problem(),
        compile_options=xip.CompileOptions(derivatives=xip.Derivatives.SPARSE),
        constraint_bounds=NORM_BOUNDS,
    )
    assert solution.status.succeeded
    assert solution.state is not None
    np.testing.assert_allclose(
        solution.state, np.sqrt(np.sum(np.asarray(solution.x) ** 2) + 1.0), atol=1e-5
    )


def test_native_layer_checks_the_initial_state() -> None:
    """Checks that the native solve refuses a missing or an unexpected initial state."""
    stateful = xip.compile(counting_problem())
    stateless = xip.compile(stateless_norm_problem())
    with pytest.raises(ValueError, match="needs an initial state"):
        stateful.native.solve(
            np.zeros(2),
            np.zeros(1),
            np.zeros(2),
            np.full(2, 3.0),
            np.zeros(0),
            np.zeros(0),
        )
    with pytest.raises(ValueError, match="takes none"):
        stateless.native.solve(
            np.zeros(2),
            np.zeros(1),
            np.full(2, -np.inf),
            np.full(2, np.inf),
            np.full(1, -np.inf),
            np.full(1, 2.0),
            initial_state=np.zeros(1),
        )
