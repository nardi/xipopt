"""Solving compiled problems with Ipopt."""

import math
import time
from typing import Any, NamedTuple

import cyxla
import numpy as np
from jaxtyping import PyTree

from xipopt._bounds import bounds_vectors, start_point
from xipopt._compile import CompiledProblem, compile  # noqa: A004
from xipopt._options import CompileOptions, Hessian, SolveOptions
from xipopt._problem_types import BoundsTree, Problem, StatefulProblem
from xipopt._solver import (
    IpoptValue,
    IterationState,
    Multipliers,
    NativeResult,
    set_blas_threads,
)
from xipopt._types import (
    BuildStatistics,
    CallbackStatistics,
    IterationInfo,
    Program,
    ProgramStatistics,
    Sense,
    SolveStatistics,
    Status,
)


class Solution(NamedTuple):
    """The result of solving a problem.

    The multipliers are those of the problem Ipopt solves, which minimizes the objective, so
    a problem that maximizes has the multipliers of its negated objective.
    """

    x: PyTree[np.ndarray]
    """The final point in the structure of the variables."""
    objective: float
    """The objective at `x`, in the sense of the problem."""
    status: Status
    constraints: PyTree[np.ndarray]
    """The constraint values at `x`, in the structure of `Evaluation.constraints`."""
    constraint_multipliers: PyTree[np.ndarray]
    """The multipliers of the constraints, in the structure of `Evaluation.constraints`."""
    lower_multipliers: PyTree[np.ndarray]
    """Multipliers of the lower bounds of the variables, in the structure of `x`."""
    upper_multipliers: PyTree[np.ndarray]
    """Multipliers of the upper bounds of the variables, in the structure of `x`."""
    iterations: int
    solving_seconds: float
    statistics: SolveStatistics
    """Counters of the callbacks that Ipopt made during the solve."""
    build: BuildStatistics
    """Time and counts of compiling the problem, zero if it was compiled before."""
    state: PyTree[np.ndarray] = None
    """The state of a `StatefulProblem` after the last evaluation, or `None` without state.

    Passing the solution as `warm_start` continues from this state.
    """


def solve(
    problem: CompiledProblem | Problem | StatefulProblem,
    x0: PyTree[Any] = None,
    params: PyTree[Any] = None,
    options: SolveOptions = SolveOptions(),
    compile_options: CompileOptions = CompileOptions(),
    warm_start: Solution | None = None,
    *,
    variable_bounds: BoundsTree | None = None,
    constraint_bounds: BoundsTree | None = None,
) -> Solution:
    """Solves a problem for the given parameter values and bounds, starting from `x0`.

    A `Problem` is compiled first with `compile_options`, while a `CompiledProblem` ignores
    them. `variable_bounds` and `constraint_bounds` are a `Bounds`, or a pytree with the
    structure of the variables or of the constraints and a `Bounds` at every leaf. Without
    `variable_bounds` the variables are unbounded, and a problem with constraints needs
    `constraint_bounds`. The bounds are read when the solve starts, so one compiled problem
    can be solved for many sets of bounds. Without `x0` the start is the point nearest to
    the origin inside the bounds. A previous `Solution` as `warm_start` supplies the
    multipliers that Ipopt starts from, and `x0` should then be its `x`. A problem with
    state starts from the state of the `warm_start` when it has one, and from the initial
    state of the problem otherwise.
    """
    build = BuildStatistics()
    if isinstance(problem, Problem | StatefulProblem):
        problem = compile(problem, params, compile_options)
        build = problem.statistics
    variables = problem.variables
    variable_lower, variable_upper = bounds_vectors(
        variable_bounds, variables, "variable_bounds", "variables"
    )
    if (constraint_bounds is None) != (problem.constraints.size == 0):
        raise ValueError(
            "`constraint_bounds` is required for a problem with constraints and not "
            "allowed for one without."
        )
    constraint_lower, constraint_upper = bounds_vectors(
        constraint_bounds, problem.constraints, "constraint_bounds", "constraints"
    )
    start = (
        start_point(variable_lower, variable_upper)
        if x0 is None
        else variables.flatten(x0, "starting point")
    )
    multipliers = (
        None if warm_start is None else warm_start_multipliers(problem, warm_start)
    )
    sign = -1.0 if problem.sense is Sense.MAXIMIZE else 1.0

    observer = None
    if options.callback is not None:
        callback = options.callback

        def observer(state: IterationState) -> bool:
            info = IterationInfo(
                iteration=state.iteration,
                restoration=state.mode == 1,
                objective=sign * state.objective,
                primal_infeasibility=state.primal_infeasibility,
                dual_infeasibility=state.dual_infeasibility,
                barrier_parameter=state.barrier_parameter,
                step_norm=state.step_norm,
                regularization_size=state.regularization_size,
                dual_step_size=state.dual_step_size,
                primal_step_size=state.primal_step_size,
                line_search_trials=state.line_search_trials,
                x=variables.unflatten(state.variables),
            )
            return callback(info) is not False

    if options.blas_threads is not None:
        set_blas_threads(options.blas_threads)
    errors_before = cyxla.error_info().count
    begin = time.perf_counter()
    result = problem.native.solve(
        start,
        problem.params.flatten(params),
        variable_lower,
        variable_upper,
        constraint_lower,
        constraint_upper,
        ipopt_settings(problem, options, warm_start is not None),
        multipliers,
        observer,
        starting_state(problem, warm_start),
    )
    seconds = time.perf_counter() - begin
    status = Status(result.status)
    errors = cyxla.error_info()
    if errors.count > errors_before and not status.succeeded:
        raise RuntimeError(
            f"A program failed during the solve, and Ipopt stopped with {status.name}: "
            f"{errors.last_message}"
        )
    return unpack(problem, result, status, sign, seconds, build)


def starting_state(
    problem: CompiledProblem, warm_start: Solution | None
) -> np.ndarray | None:
    """Returns the flat state that a solve starts from, or `None` for a problem without state."""
    layout = problem.state
    if layout is None:
        return None
    if warm_start is not None and warm_start.state is not None:
        return layout.flatten(warm_start.state)
    return problem.initial_state


def ipopt_settings(
    problem: CompiledProblem, options: SolveOptions, warm_started: bool
) -> list[tuple[str, IpoptValue]]:
    """Returns the Ipopt options of a solve, with the settings of the caller last."""
    settings: list[tuple[str, IpoptValue]] = [
        ("tol", float(options.tolerance)),
        ("max_iter", int(options.max_iterations)),
        ("print_level", int(options.print_level)),
        ("linear_solver", options.linear_solver.value),
    ]
    if options.time_limit is not None and math.isfinite(options.time_limit):
        settings.append(("max_wall_time", float(options.time_limit)))
    if problem.options.hessian is Hessian.LIMITED_MEMORY:
        settings.append(("hessian_approximation", "limited-memory"))
    if warm_started:
        settings.append(("warm_start_init_point", "yes"))
    settings.extend((setting.option, setting.value) for setting in options.settings)
    return settings


def warm_start_multipliers(problem: CompiledProblem, solution: Solution) -> Multipliers:
    """Returns the multipliers of a solution as the flat vectors that Ipopt starts from."""
    return Multipliers(
        problem.constraints.flatten(
            solution.constraint_multipliers, "constraint multipliers"
        ),
        problem.variables.flatten(solution.lower_multipliers, "lower multipliers"),
        problem.variables.flatten(solution.upper_multipliers, "upper multipliers"),
    )


def solve_statistics(result: NativeResult) -> SolveStatistics:
    """Returns the counters of a solve from the counters of the native layer."""

    def callback(index: int) -> CallbackStatistics:
        return CallbackStatistics(
            result.callback_calls[index], result.callback_hits[index]
        )

    def program(which: Program) -> ProgramStatistics:
        return ProgramStatistics(result.executions[which], result.seconds[which])

    return SolveStatistics(
        objective=callback(0),
        gradient=callback(1),
        constraints=callback(2),
        jacobian=callback(3),
        hessian=callback(4),
        first_order_program=program(Program.FIRST_ORDER),
        hessian_program=program(Program.HESSIAN),
        unused_derivatives=result.unused_derivatives,
    )


def unpack(
    problem: CompiledProblem,
    result: NativeResult,
    status: Status,
    sign: float,
    seconds: float,
    build: BuildStatistics,
) -> Solution:
    """Returns a `Solution` from the flat vectors of the native layer."""
    return Solution(
        x=problem.variables.unflatten(result.variables),
        objective=sign * result.objective,
        status=status,
        constraints=problem.constraints.unflatten(result.constraints),
        constraint_multipliers=problem.constraints.unflatten(
            result.constraint_multipliers
        ),
        lower_multipliers=problem.variables.unflatten(result.lower_multipliers),
        upper_multipliers=problem.variables.unflatten(result.upper_multipliers),
        iterations=result.iterations,
        solving_seconds=seconds,
        statistics=solve_statistics(result),
        build=build,
        state=None if problem.state is None else problem.state.unflatten(result.state),
    )
