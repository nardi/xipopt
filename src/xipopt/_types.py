"""Enumerations and records shared by the extension module and the Python layer."""

from enum import IntEnum
from typing import NamedTuple

import numpy as np
from jaxtyping import PyTree


class Sense(IntEnum):
    """Direction of optimization."""

    MAXIMIZE = -1
    MINIMIZE = 1


class Status(IntEnum):
    """Why Ipopt stopped, with the values of its `ApplicationReturnStatus`."""

    SOLVE_SUCCEEDED = 0
    SOLVED_TO_ACCEPTABLE_LEVEL = 1
    INFEASIBLE_PROBLEM_DETECTED = 2
    SEARCH_DIRECTION_BECOMES_TOO_SMALL = 3
    DIVERGING_ITERATES = 4
    USER_REQUESTED_STOP = 5
    FEASIBLE_POINT_FOUND = 6
    MAXIMUM_ITERATIONS_EXCEEDED = -1
    RESTORATION_FAILED = -2
    ERROR_IN_STEP_COMPUTATION = -3
    MAXIMUM_CPU_TIME_EXCEEDED = -4
    MAXIMUM_WALL_TIME_EXCEEDED = -5
    NOT_ENOUGH_DEGREES_OF_FREEDOM = -10
    INVALID_PROBLEM_DEFINITION = -11
    INVALID_OPTION = -12
    INVALID_NUMBER_DETECTED = -13
    UNRECOVERABLE_EXCEPTION = -100
    NON_IPOPT_EXCEPTION_THROWN = -101
    INSUFFICIENT_MEMORY = -102
    INTERNAL_ERROR = -199

    @property
    def succeeded(self) -> bool:
        """Whether Ipopt found a point that satisfies the tolerances or the acceptable ones."""
        return self in (
            Status.SOLVE_SUCCEEDED,
            Status.SOLVED_TO_ACCEPTABLE_LEVEL,
            Status.FEASIBLE_POINT_FOUND,
        )


class Program(IntEnum):
    """The programs that Ipopt's callbacks execute, in the order the native layer indexes them."""

    FIRST_ORDER = 0
    """The objective, the gradient, the constraints and the Jacobian in one pass."""
    HESSIAN = 1
    """The Hessian of the Lagrangian."""


class CallbackStatistics(NamedTuple):
    """Counters of one callback of Ipopt."""

    calls: int
    """Times Ipopt called the callback."""
    cache_hits: int
    """Calls that were answered from an execution that an earlier callback at the same point
    had caused, so they ran no program."""


class ProgramStatistics(NamedTuple):
    """Counters of one program."""

    executions: int
    """Times the program ran."""
    seconds: float
    """Seconds spent executing the program."""


class SolveStatistics(NamedTuple):
    """Counters of every callback of Ipopt and of both programs for one solve."""

    objective: CallbackStatistics
    gradient: CallbackStatistics
    constraints: CallbackStatistics
    jacobian: CallbackStatistics
    hessian: CallbackStatistics
    first_order_program: ProgramStatistics
    """The program for the objective, the gradient, the constraints and the Jacobian, which
    runs once for each new point at which one of those callbacks evaluates a value."""
    hessian_program: ProgramStatistics
    unused_derivatives: int
    """Points at which the gradient and the Jacobian were computed with the values, and
    Ipopt never asked for them. Ipopt evaluates trial points that it rejects."""


class BuildStatistics(NamedTuple):
    """Time and counts of building the programs of a problem."""

    detect_seconds: float = 0.0
    """Seconds spent detecting and coloring sparsity patterns."""
    lower_seconds: float = 0.0
    """Seconds spent tracing and lowering programs to StableHLO."""
    compile_seconds: float = 0.0
    """Seconds spent compiling StableHLO with the PJRT plugin."""
    load_seconds: float = 0.0
    """Seconds spent reading lowered programs and executables from the disk cache."""
    lowered: int = 0
    """Programs that were lowered."""
    compiled: int = 0
    """Programs that were compiled."""
    loaded: int = 0
    """Programs whose executables came from the disk cache."""
    wall_seconds: float = 0.0
    """Seconds the build took on the wall clock."""


class IterationInfo(NamedTuple):
    """What Ipopt reports at the end of an iteration, passed to the callback of a solve."""

    iteration: int
    restoration: bool
    """Whether Ipopt is in its restoration phase."""
    objective: float
    primal_infeasibility: float
    dual_infeasibility: float
    barrier_parameter: float
    step_norm: float
    """Infinity norm of the primal step."""
    regularization_size: float
    """Size of the regularization of the Hessian of the Lagrangian."""
    dual_step_size: float
    primal_step_size: float
    line_search_trials: int
    x: PyTree[np.ndarray]
    """The current iterate in the structure of the variables."""
