"""Tests of the options, the callback and the warm start of a solve."""

import sys
import threading

import jax.numpy as jnp
import numpy as np
import pytest
from problems import bratu_control, hs071

import xipopt as xip


@pytest.fixture(scope="module")
def compiled() -> xip.CompiledProblem:
    """Returns problem 71 compiled."""
    return xip.compile(hs071().problem)


START = hs071().start


def test_options_reach_ipopt(compiled: xip.CompiledProblem) -> None:
    """Checks that the fields of the options change the solve."""
    loose = xip.solve(compiled, START, options=xip.SolveOptions(tolerance=1e-2))
    tight = xip.solve(compiled, START, options=xip.SolveOptions(tolerance=1e-12))
    assert loose.iterations < tight.iterations
    limited = xip.solve(compiled, START, options=xip.SolveOptions(max_iterations=2))
    assert limited.status is xip.Status.MAXIMUM_ITERATIONS_EXCEEDED
    assert not limited.status.succeeded
    assert limited.iterations == 2


def test_time_limit_stops_a_solve() -> None:
    """Checks that the wall time limit ends a solve that needs many iterations."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
    )
    solution = xip.solve(
        problem,
        np.array([-1.2, 1.0]),
        options=xip.SolveOptions(time_limit=1e-9, tolerance=1e-14),
    )
    assert solution.status is xip.Status.MAXIMUM_WALL_TIME_EXCEEDED


def test_infinite_time_limit_is_no_limit(compiled: xip.CompiledProblem) -> None:
    """Checks that an infinite limit is left out instead of passed to Ipopt."""
    solution = xip.solve(
        compiled, START, options=xip.SolveOptions(time_limit=float("inf"))
    )
    assert solution.status.succeeded


@pytest.mark.parametrize("linear_solver", list(xip.LinearSolver))
def test_linear_solvers(
    compiled: xip.CompiledProblem, linear_solver: xip.LinearSolver
) -> None:
    """Checks that both sparse solvers of the wheel solve the problem."""
    solution = xip.solve(
        compiled, START, options=xip.SolveOptions(linear_solver=linear_solver)
    )
    assert solution.status.succeeded
    assert solution.objective == pytest.approx(17.0140173, abs=1e-6)


def test_settings_of_the_caller_come_last(
    compiled: xip.CompiledProblem, capfd: pytest.CaptureFixture[str]
) -> None:
    """Checks that settings override the fields and print through Ipopt."""
    options = xip.SolveOptions(
        settings=(
            xip.IpoptSetting("print_level", 5),
            xip.IpoptSetting("mu_strategy", "adaptive"),
            xip.IpoptSetting("acceptable_tol", 1e-4),
        ),
        blas_threads=None,
    )
    xip.solve(compiled, START, options=options)
    # Ipopt buffers its output in the C++ runtime of Windows until the process ends.
    if sys.platform != "win32":
        assert "iter" in capfd.readouterr().out


def test_rejected_settings_raise(compiled: xip.CompiledProblem) -> None:
    """Checks the errors for unknown options, wrong values and wrong types."""
    with pytest.raises(ValueError, match="rejected the option no_such_option"):
        xip.solve(
            compiled,
            START,
            options=xip.SolveOptions(settings=(xip.IpoptSetting("no_such_option", 1),)),
        )
    with pytest.raises(ValueError, match="rejected the option mu_strategy"):
        xip.solve(
            compiled,
            START,
            options=xip.SolveOptions(settings=(xip.IpoptSetting("mu_strategy", "x"),)),
        )
    for value in (True, None, (1,)):
        with pytest.raises(TypeError, match="must be a string, an integer or a float"):
            xip.solve(
                compiled,
                START,
                options=xip.SolveOptions(
                    settings=(xip.IpoptSetting("max_iter", value),)  # ty: ignore[invalid-argument-type]
                ),
            )


def test_callback_sees_every_iteration(compiled: xip.CompiledProblem) -> None:
    """Checks the information that the callback receives."""
    seen: list[xip.IterationInfo] = []
    solution = xip.solve(
        compiled, START, options=xip.SolveOptions(callback=seen.append)
    )
    assert [info.iteration for info in seen] == list(range(solution.iterations + 1))
    last = seen[-1]
    assert last.objective == pytest.approx(solution.objective, abs=1e-5)
    assert np.shape(last.x) == (4,)
    assert not last.restoration
    assert last.primal_infeasibility < 1e-6
    assert last.barrier_parameter < 1e-6
    assert last.step_norm >= 0.0
    assert last.line_search_trials >= 0
    assert last.primal_step_size <= 1.0
    assert last.dual_step_size <= 1.0
    assert last.dual_infeasibility < 1e-4
    assert last.regularization_size >= 0.0


def test_callback_stops_the_solve(compiled: xip.CompiledProblem) -> None:
    """Checks that returning `False` stops Ipopt with the matching status."""
    solution = xip.solve(
        compiled,
        START,
        options=xip.SolveOptions(callback=lambda info: info.iteration < 2),
    )
    assert solution.status is xip.Status.USER_REQUESTED_STOP
    assert solution.iterations == 2


def test_callback_exceptions_propagate(compiled: xip.CompiledProblem) -> None:
    """Checks that an exception of the callback ends the solve and is raised."""

    def callback(info: xip.IterationInfo) -> None:
        raise KeyError("stop here")

    with pytest.raises(KeyError, match="stop here"):
        xip.solve(compiled, START, options=xip.SolveOptions(callback=callback))
    # The problem solves again afterwards.
    assert xip.solve(compiled, START).status.succeeded


def test_callback_maximization_reports_the_objective_of_the_problem() -> None:
    """Checks that the callback sees the objective in the sense of the problem."""
    problem = xip.Problem(
        variables=xip.Variable(lower=-2.0, upper=2.0),
        objective=lambda x, params: 3.0 - (x - 0.5) ** 2,
        sense=xip.Sense.MAXIMIZE,
    )
    seen: list[xip.IterationInfo] = []
    xip.solve(problem, options=xip.SolveOptions(callback=seen.append))
    assert seen[-1].objective == pytest.approx(3.0)


def test_warm_start_needs_fewer_iterations() -> None:
    """Checks that the multipliers of a solution start a similar solve closer to the answer."""
    compiled = xip.compile(bratu_control(20))
    cold = xip.solve(compiled)
    warm = xip.solve(
        compiled,
        cold.x,
        options=xip.SolveOptions(
            settings=(
                xip.IpoptSetting("warm_start_bound_push", 1e-9),
                xip.IpoptSetting("mu_init", 1e-9),
            )
        ),
        warm_start=cold,
    )
    assert warm.status.succeeded
    assert warm.iterations < cold.iterations
    assert warm.objective == pytest.approx(cold.objective, rel=1e-6)


def test_warm_start_checks_the_solution(compiled: xip.CompiledProblem) -> None:
    """Checks the error for a solution that belongs to another problem."""
    other = xip.solve(
        xip.Problem(
            variables=xip.Variable(shape=(4,)),
            objective=lambda x, params: jnp.sum(x**2),
        )
    )
    with pytest.raises(ValueError, match="different number of constraints"):
        xip.solve(compiled, START, warm_start=other)


def test_concurrent_solves_are_serialized(compiled: xip.CompiledProblem) -> None:
    """Checks that solves from several threads do not disturb each other."""
    results: list[float] = []

    def work() -> None:
        results.append(xip.solve(compiled, START).objective)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [pytest.approx(17.0140173, abs=1e-6)] * 4


def test_wrong_inputs_are_reported(compiled: xip.CompiledProblem) -> None:
    """Checks the errors for a starting point and parameters of the wrong layout."""
    with pytest.raises(ValueError, match="values have structure"):
        xip.solve(compiled, {"a": START})
    with pytest.raises(ValueError, match="A value has shape"):
        xip.solve(compiled, np.zeros(3))
    with pytest.raises(ValueError, match="parameters have structure"):
        xip.solve(compiled, START, params=np.zeros(2))


def test_failing_programs_are_raised(
    compiled: xip.CompiledProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks that a failed program is raised when it ends a solve without success."""
    import cyxla

    calls = iter(range(100))
    monkeypatch.setattr(
        cyxla, "error_info", lambda: cyxla.ErrorInfo(next(calls), "the plugin failed")
    )
    with pytest.raises(RuntimeError, match="the plugin failed"):
        xip.solve(compiled, START, options=xip.SolveOptions(max_iterations=1))
    # A failure during a solve that succeeds is not an error.
    assert xip.solve(compiled, START).status.succeeded
