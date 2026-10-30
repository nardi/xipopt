"""Solving compiled problems with Ipopt."""

import math
import time
from typing import Any, NamedTuple

import cyxla
import numpy as np
from jaxtyping import PyTree

from xipopt._compile import CompiledProblem, compile  # noqa: A004
from xipopt._options import CompileOptions, Hessian, SolveOptions
from xipopt._problem_types import Problem
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
    constraints: tuple[np.ndarray, ...]
    """The value of each constraint at `x`, in its own shape."""
    constraint_multipliers: tuple[np.ndarray, ...]
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


def solve(
    problem: CompiledProblem | Problem,
    x0: PyTree[Any] = None,
    params: PyTree[Any] = None,
    options: SolveOptions = SolveOptions(),
    compile_options: CompileOptions = CompileOptions(),
    warm_start: Solution | None = None,
) -> Solution:
    """Solves a problem for the given parameter values, starting from `x0`.

    A `Problem` is compiled first with `compile_options`, while a `CompiledProblem` ignores
    them. Without `x0` the start is the point nearest to the origin inside the bounds.
    A previous `Solution` as `warm_start` supplies the multipliers that Ipopt starts from,
    and `x0` should then be its `x`.
    """
    build = BuildStatistics()
    if isinstance(problem, Problem):
        problem = compile(problem, params, compile_options)
        build = problem.statistics
    variables = problem.variables
    start = variables.start() if x0 is None else variables.flatten(x0)
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
        ipopt_settings(problem, options, warm_start is not None),
        multipliers,
        observer,
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
    if len(solution.constraint_multipliers) != len(problem.constraint_shapes):
        raise ValueError(
            "The solution to warm start from has a different number of constraints."
        )
    parts = [
        np.asarray(part, np.float64).ravel() for part in solution.constraint_multipliers
    ]
    return Multipliers(
        np.concatenate(parts) if parts else np.zeros(0),
        problem.variables.flatten(solution.lower_multipliers),
        problem.variables.flatten(solution.upper_multipliers),
    )


def split_constraints(
    problem: CompiledProblem, values: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Returns a vector over all constraints as one array per constraint."""
    arrays = []
    offset = 0
    for shape in problem.constraint_shapes:
        size = math.prod(shape)
        arrays.append(values[offset : offset + size].reshape(shape))
        offset += size
    return tuple(arrays)


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
        constraints=split_constraints(problem, result.constraints),
        constraint_multipliers=split_constraints(
            problem, result.constraint_multipliers
        ),
        lower_multipliers=problem.variables.unflatten(result.lower_multipliers),
        upper_multipliers=problem.variables.unflatten(result.upper_multipliers),
        iterations=result.iterations,
        solving_seconds=seconds,
        statistics=SolveStatistics(
            *(
                CallbackStatistics(result.calls[program], result.seconds[program])
                for program in Program
            )
        ),
        build=build,
    )
