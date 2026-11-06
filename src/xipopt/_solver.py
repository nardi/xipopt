"""Ipopt callbacks that execute cyxla programs without the GIL.

A `NativeProblem` holds the compiled programs of one problem together with the buffers that
they read. Ipopt's callbacks copy its arrays into those buffers, execute a program and write
the result into Ipopt's output arrays, so a solve never enters the interpreter unless the
caller asked to observe the iterations.
"""

import threading
from collections.abc import Callable, Sequence
from typing import NamedTuple

import cython
import cyxla
import numpy as np
from cython.cimports.cyxla import (
    ABI_VERSION,
    MAX_OUTPUTS,
    Executable,
    PJRT_Buffer_Type_F64,
    alias_buffer,
    aligned_memory,
    destroy_buffer,
    execute,
    free_aligned,
    now,
)
from cython.cimports.libc.math import isfinite
from cython.cimports.libc.stdlib import calloc, free, malloc
from cython.cimports.libc.string import memcmp, memcpy
from cython.cimports.xipopt._blas import (
    lbt_forward,
    openblas_set_num_threads,
)
from cython.cimports.xipopt._ipopt import (
    AddIpoptIntOption,
    AddIpoptNumOption,
    AddIpoptStrOption,
    CreateIpoptProblem,
    FreeIpoptProblem,
    GetIpoptCurrentIterate,
    GetIpoptVersion,
    IpoptProblem,
    IpoptSolve,
    SetIntermediateCallback,
    ipbool,
    ipindex,
    ipnumber,
)
from cython.cimports.xipopt._platform import xipopt_library_path
from cython.cimports.xipopt._solver import (
    CALLBACK_COUNT,
    CONSTRAINTS_CALLBACK,
    FIRST_ORDER,
    GRADIENT_CALLBACK,
    HESSIAN,
    HESSIAN_CALLBACK,
    INPUT_COUNT,
    JACOBIAN_CALLBACK,
    MULTIPLIERS,
    OBJECTIVE_CALLBACK,
    OBJECTIVE_FACTOR,
    PARAMETERS,
    PROGRAM_COUNT,
    VARIABLES,
    SolverState,
)

from xipopt._environment import library_path

cyxla.check_abi(ABI_VERSION)

SOLVE_LOCK = threading.Lock()
"""Serializes solves, because the runtime of cyxla and its error buffer are not thread safe."""

IpoptValue = str | int | float
"""A value of an Ipopt option, which the C interface sets by its type."""


class SolverError(RuntimeError):
    """An error in the native layer, such as an option that Ipopt rejects."""


class IterationState(NamedTuple):
    """What Ipopt reports at the end of an iteration."""

    mode: int
    """Algorithm mode, 0 for the regular algorithm and 1 for the restoration phase."""
    iteration: int
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
    variables: np.ndarray
    """The current iterate as a flat vector."""


class Multipliers(NamedTuple):
    """The multipliers of a solve, which also start a warm started solve."""

    constraints: np.ndarray
    lower: np.ndarray
    """Multipliers of the lower bounds of the variables."""
    upper: np.ndarray
    """Multipliers of the upper bounds of the variables."""


class NativeResult(NamedTuple):
    """The outcome of one solve in the units of Ipopt."""

    status: int
    """Value of Ipopt's `ApplicationReturnStatus`."""
    objective: float
    variables: np.ndarray
    constraints: np.ndarray
    constraint_multipliers: np.ndarray
    lower_multipliers: np.ndarray
    upper_multipliers: np.ndarray
    iterations: int
    callback_calls: tuple[int, ...]
    """Calls of each callback of Ipopt: objective, gradient, constraints, Jacobian, Hessian."""
    callback_hits: tuple[int, ...]
    """Calls of each callback that were answered from the cache and ran no program."""
    executions: tuple[int, ...]
    """Executions of each program, first-order and Hessian."""
    seconds: tuple[float, ...]
    """Seconds spent executing each program."""
    unused_derivatives: int
    """Points at which the first-order program computed derivatives that Ipopt never asked for."""
    state: np.ndarray
    """The state after the last successful execution of the first-order program, empty for a
    problem without state."""


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def run_program(
    state: cython.pointer[SolverState], program: cython.int, destination: cython.p_void
) -> ipbool:
    """Executes the Hessian program into `destination` and returns whether it succeeded.

    A program that was not compiled counts as success, since there is nothing to compute.
    """
    if state.programs[program].handle == cython.NULL:
        return 1
    outputs: cython.p_void[MAX_OUTPUTS]
    outputs[0] = destination
    start: cython.double = now()
    status: cython.int = execute(
        cython.address(state.programs[program]),
        state.inputs,
        outputs,
        state.zero_copy,
    )
    state.executions[program] += 1
    state.seconds[program] += now() - start
    return status == 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def all_finite(values: cython.pointer[cython.double], size: cython.int) -> ipbool:
    """Returns whether none of the values is infinite or not a number."""
    index: cython.int
    for index in range(size):
        if not isfinite(values[index]):
            return 0
    return 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def execute_first_order(
    state: cython.pointer[SolverState], x: cython.pointer[ipnumber]
) -> ipbool:
    """Runs the first-order program at `x` and keeps its outputs in the cache.

    The point counts as failed when the program fails or when any output is infinite or not
    a number. Ipopt then shortens its step if it asked for a value at a trial point, as it
    does for any point where the functions cannot be evaluated.
    """
    n: cython.int = state.variable_count
    # The derivatives that the previous point computed and Ipopt never asked for are wasted.
    if state.cache_valid and not state.cache_failed and not state.derivatives_requested:
        state.unused_derivatives += 1
    memcpy(state.inputs[VARIABLES].host, x, n * cython.sizeof(ipnumber))
    memcpy(state.cached_x, x, n * cython.sizeof(ipnumber))
    outputs: cython.p_void[MAX_OUTPUTS]
    outputs[0] = state.objective_value
    outputs[1] = state.gradient_values
    outputs[2] = state.constraint_values if state.constraint_count > 0 else cython.NULL
    outputs[3] = state.jacobian_values if state.jacobian_size > 0 else cython.NULL
    outputs[STATE_OUTPUT] = state.state_values if state.state_size > 0 else cython.NULL
    start: cython.double = now()
    status: cython.int = execute(
        cython.address(state.programs[FIRST_ORDER]),
        state.inputs,
        outputs,
        state.zero_copy,
    )
    state.executions[FIRST_ORDER] += 1
    state.seconds[FIRST_ORDER] += now() - start
    state.cache_valid = 1
    state.derivatives_requested = 0
    state.cache_failed = (
        status != 0
        or not all_finite(state.objective_value, 1)
        or not all_finite(state.gradient_values, n)
        or not all_finite(state.constraint_values, state.constraint_count)
        or not all_finite(state.jacobian_values, state.jacobian_size)
        or not all_finite(state.state_values, state.state_size)
    )
    if state.state_size > 0 and not state.cache_failed:
        # The next evaluation starts from the state that this one ended in. A failed
        # evaluation leaves the state as it was.
        memcpy(
            cython.cast(cython.pointer[ipnumber], state.inputs[PARAMETERS].host)
            + state.parameter_count,
            state.state_values,
            state.state_size * cython.sizeof(ipnumber),
        )
    return not state.cache_failed


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def serve(
    state: cython.pointer[SolverState],
    callback: cython.int,
    x: cython.pointer[ipnumber],
) -> ipbool:
    """Stores the outputs at `x` in the cache, counts the call and returns whether they are usable.

    The point is compared bit by bit with the one in the cache and not through Ipopt's
    `new_x` flag, since the restoration phase and rejected trial points make that flag a
    poor guide to whether the point changed.
    """
    state.callback_calls[callback] += 1
    if state.cache_valid and (
        memcmp(state.cached_x, x, state.variable_count * cython.sizeof(ipnumber)) == 0
    ):
        state.callback_hits[callback] += 1
    else:
        execute_first_order(state, x)
    return not state.cache_failed


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def evaluate_objective(
    n: ipindex,
    x: cython.pointer[ipnumber],
    new_x: ipbool,
    objective: cython.pointer[ipnumber],
    user_data: cython.p_void,
) -> ipbool:
    state: cython.pointer[SolverState] = cython.cast(
        cython.pointer[SolverState], user_data
    )
    if not serve(state, OBJECTIVE_CALLBACK, x):
        return 0
    objective[0] = state.objective_value[0]
    return 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def evaluate_gradient(
    n: ipindex,
    x: cython.pointer[ipnumber],
    new_x: ipbool,
    gradient: cython.pointer[ipnumber],
    user_data: cython.p_void,
) -> ipbool:
    state: cython.pointer[SolverState] = cython.cast(
        cython.pointer[SolverState], user_data
    )
    if not serve(state, GRADIENT_CALLBACK, x):
        return 0
    state.derivatives_requested = 1
    memcpy(gradient, state.gradient_values, n * cython.sizeof(ipnumber))
    return 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def evaluate_constraints(
    n: ipindex,
    x: cython.pointer[ipnumber],
    new_x: ipbool,
    m: ipindex,
    constraints: cython.pointer[ipnumber],
    user_data: cython.p_void,
) -> ipbool:
    state: cython.pointer[SolverState] = cython.cast(
        cython.pointer[SolverState], user_data
    )
    if not serve(state, CONSTRAINTS_CALLBACK, x):
        return 0
    memcpy(constraints, state.constraint_values, m * cython.sizeof(ipnumber))
    return 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def evaluate_jacobian(
    n: ipindex,
    x: cython.pointer[ipnumber],
    new_x: ipbool,
    m: ipindex,
    size: ipindex,
    rows: cython.pointer[ipindex],
    columns: cython.pointer[ipindex],
    values: cython.pointer[ipnumber],
    user_data: cython.p_void,
) -> ipbool:
    state: cython.pointer[SolverState] = cython.cast(
        cython.pointer[SolverState], user_data
    )
    if values == cython.NULL:
        # Ipopt asks for the structure of the Jacobian once, before the first evaluation.
        memcpy(rows, state.jacobian_rows, size * cython.sizeof(ipindex))
        memcpy(columns, state.jacobian_columns, size * cython.sizeof(ipindex))
        return 1
    if not serve(state, JACOBIAN_CALLBACK, x):
        return 0
    state.derivatives_requested = 1
    memcpy(values, state.jacobian_values, size * cython.sizeof(ipnumber))
    return 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def evaluate_hessian(
    n: ipindex,
    x: cython.pointer[ipnumber],
    new_x: ipbool,
    objective_factor: ipnumber,
    m: ipindex,
    multipliers: cython.pointer[ipnumber],
    new_multipliers: ipbool,
    size: ipindex,
    rows: cython.pointer[ipindex],
    columns: cython.pointer[ipindex],
    values: cython.pointer[ipnumber],
    user_data: cython.p_void,
) -> ipbool:
    state: cython.pointer[SolverState] = cython.cast(
        cython.pointer[SolverState], user_data
    )
    if values == cython.NULL:
        # Ipopt asks for the structure even when it approximates the Hessian, in which case
        # the structure is empty and the values are never requested.
        memcpy(rows, state.hessian_rows, size * cython.sizeof(ipindex))
        memcpy(columns, state.hessian_columns, size * cython.sizeof(ipindex))
        return 1
    state.callback_calls[HESSIAN_CALLBACK] += 1
    memcpy(state.inputs[VARIABLES].host, x, n * cython.sizeof(ipnumber))
    cython.cast(cython.pointer[ipnumber], state.inputs[OBJECTIVE_FACTOR].host)[0] = (
        objective_factor
    )
    memcpy(state.inputs[MULTIPLIERS].host, multipliers, m * cython.sizeof(ipnumber))
    return run_program(state, HESSIAN, values)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def report_iteration(
    mode: ipindex,
    iteration: ipindex,
    objective: ipnumber,
    primal_infeasibility: ipnumber,
    dual_infeasibility: ipnumber,
    barrier_parameter: ipnumber,
    step_norm: ipnumber,
    regularization_size: ipnumber,
    dual_step_size: ipnumber,
    primal_step_size: ipnumber,
    line_search_trials: ipindex,
    user_data: cython.p_void,
) -> ipbool:
    """Counts the iteration and, when a solve observes them, calls the observer with the GIL."""
    state: cython.pointer[SolverState] = cython.cast(
        cython.pointer[SolverState], user_data
    )
    state.iterations = max(state.iterations, iteration)
    if state.observer == cython.NULL:
        return 1
    with cython.gil:
        problem: NativeProblem = cython.cast(NativeProblem, state.observer)
        return problem.notify(
            mode,
            iteration,
            objective,
            primal_infeasibility,
            dual_infeasibility,
            barrier_parameter,
            step_norm,
            regularization_size,
            dual_step_size,
            primal_step_size,
            line_search_trials,
        )


def forward_blas() -> int:
    """Forwards the linked OpenBLAS to libblastrampoline and returns the number of forwarded symbols.

    Ipopt calls BLAS and LAPACK through libblastrampoline, which has no backend of its own.
    The library path comes from the loaded OpenBLAS, so it does not depend on environment
    variables or install locations.
    """
    address: cython.p_void = cython.cast(cython.p_void, openblas_set_num_threads)
    path: cython.p_char = cython.cast(cython.p_char, xipopt_library_path(address))
    # The path of the library has to outlive the call, so it is kept in a local variable.
    fallback: bytes = b""
    if path == cython.NULL:  # pragma: no cover
        found = library_path("libopenblas.dll")  # pragma: no cover
        if found is not None:  # pragma: no cover
            fallback = found  # pragma: no cover
            path = fallback  # pragma: no cover
    if path == cython.NULL:
        raise SolverError(  # pragma: no cover
            "The path of the linked OpenBLAS is unknown."
        )
    count: cython.int = lbt_forward(path, 1, 0, cython.NULL)
    if count <= 0:
        raise SolverError(  # pragma: no cover
            "libblastrampoline could not forward the linked OpenBLAS."
        )
    return count


def set_blas_threads(count: int) -> None:
    """Sets the number of threads that the linked OpenBLAS uses."""
    openblas_set_num_threads(count)


def ipopt_version() -> tuple[int, int, int]:
    """Returns the major, minor and patch version of the linked Ipopt."""
    major: cython.int = 0
    minor: cython.int = 0
    patch: cython.int = 0
    GetIpoptVersion(cython.address(major), cython.address(minor), cython.address(patch))
    return (major, minor, patch)


@cython.cfunc
def float_array(values: object, size: cython.Py_ssize_t, name: str) -> object:
    """Returns a copy of `values` as a contiguous float64 vector of the given size."""
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    if array.shape != (size,):
        raise ValueError(f"The {name} must have shape ({size},), not {array.shape}.")
    return array


@cython.cclass
class NativeProblem:
    """The programs and index arrays of a problem, with the buffers their callbacks share.

    `executables` has the first-order program and the Hessian of the Lagrangian, where
    `None` leaves out the Hessian. Programs read the variables and the parameters, followed
    by the state if the problem has one, and the Hessian program also the objective factor
    and the constraint multipliers. The first-order program writes the objective, the
    gradient, the constraints and the Jacobian values, and the new state if there is one.
    The callbacks that evaluate them at one point share one execution. The Jacobian and
    Hessian entries follow the given row and column indices, and the Hessian contains the
    lower triangle only.
    """

    state: cython.pointer[SolverState]
    executables = cython.declare(tuple, visibility="readonly")
    """The executables, which have to outlive the handles that the state holds."""
    current_observer: object
    observer_error: object

    def __cinit__(
        self,
        executables: Sequence[Executable | None],
        variable_count: cython.int,
        constraint_count: cython.int,
        parameter_size: cython.int,
        jacobian_rows: object,
        jacobian_columns: object,
        hessian_rows: object,
        hessian_columns: object,
        zero_copy: cython.bint = True,
        state_size: cython.int = 0,
    ) -> None:
        self.state = cython.NULL
        self.executables = tuple(executables)
        self.current_observer = None
        self.observer_error = None
        if len(self.executables) != PROGRAM_COUNT:
            raise ValueError(f"Expected {PROGRAM_COUNT} programs.")
        padded: cython.int = max(constraint_count, 1)

        jacobian_size: cython.int = len(jacobian_rows)
        hessian_size: cython.int = len(hessian_rows)
        state: cython.pointer[SolverState] = cython.cast(
            cython.pointer[SolverState], calloc(1, cython.sizeof(SolverState))
        )
        if state == cython.NULL:
            raise MemoryError()
        self.state = state
        state.variable_count = variable_count
        state.constraint_count = constraint_count
        state.jacobian_size = jacobian_size
        state.hessian_size = hessian_size
        state.zero_copy = zero_copy
        state.parameter_count = max(parameter_size, 1)
        state.state_size = state_size
        state.jacobian_rows = copy_indices(jacobian_rows)
        state.jacobian_columns = copy_indices(jacobian_columns)
        state.hessian_rows = copy_indices(hessian_rows)
        state.hessian_columns = copy_indices(hessian_columns)

        sizes: cython.int[INPUT_COUNT]
        sizes[VARIABLES] = variable_count
        sizes[PARAMETERS] = state.parameter_count + state_size
        sizes[OBJECTIVE_FACTOR] = 1
        sizes[MULTIPLIERS] = padded
        index: cython.int
        for index in range(INPUT_COUNT):
            state.inputs[index].type = PJRT_Buffer_Type_F64
            state.inputs[index].size = sizes[index]
            state.inputs[index].host = aligned_memory(
                sizes[index] * cython.sizeof(ipnumber)
            )
            if state.inputs[index].host == cython.NULL:
                raise MemoryError()
            state.inputs[index].buffer = alias_buffer(
                state.inputs[index].host, PJRT_Buffer_Type_F64, sizes[index]
            )

        # The buffers of the first-order outputs are padded to at least one entry, so no
        # pointer is NULL where a program may write.
        state.cached_x = cython.cast(
            cython.pointer[cython.double],
            calloc(max(variable_count, 1), cython.sizeof(ipnumber)),
        )
        state.objective_value = cython.cast(
            cython.pointer[cython.double], calloc(1, cython.sizeof(ipnumber))
        )
        state.gradient_values = cython.cast(
            cython.pointer[cython.double],
            calloc(max(variable_count, 1), cython.sizeof(ipnumber)),
        )
        state.constraint_values = cython.cast(
            cython.pointer[cython.double], calloc(padded, cython.sizeof(ipnumber))
        )
        state.jacobian_values = cython.cast(
            cython.pointer[cython.double],
            calloc(max(jacobian_size, 1), cython.sizeof(ipnumber)),
        )
        state.state_values = cython.cast(
            cython.pointer[cython.double],
            calloc(max(state_size, 1), cython.sizeof(ipnumber)),
        )
        if (
            state.state_values == cython.NULL
            or state.cached_x == cython.NULL
            or state.objective_value == cython.NULL
            or state.gradient_values == cython.NULL
            or state.constraint_values == cython.NULL
            or state.jacobian_values == cython.NULL
        ):
            raise MemoryError()

        executable: Executable
        if self.executables[FIRST_ORDER] is None:
            raise ValueError("The first-order program is required.")
        executable = self.executables[FIRST_ORDER]
        state.programs[FIRST_ORDER].handle = executable.handle
        state.programs[FIRST_ORDER].num_inputs = OBJECTIVE_FACTOR
        state.programs[FIRST_ORDER].num_outputs = 4 + (1 if state_size > 0 else 0)
        state.programs[FIRST_ORDER].output_bytes[0] = cython.sizeof(ipnumber)
        state.programs[FIRST_ORDER].output_bytes[1] = variable_count * cython.sizeof(
            ipnumber
        )
        state.programs[FIRST_ORDER].output_bytes[2] = constraint_count * cython.sizeof(
            ipnumber
        )
        state.programs[FIRST_ORDER].output_bytes[3] = jacobian_size * cython.sizeof(
            ipnumber
        )
        state.programs[FIRST_ORDER].output_bytes[STATE_OUTPUT] = (
            state_size * cython.sizeof(ipnumber)
        )
        if self.executables[HESSIAN] is not None:
            executable = self.executables[HESSIAN]
            state.programs[HESSIAN].handle = executable.handle
            state.programs[HESSIAN].num_inputs = INPUT_COUNT
            state.programs[HESSIAN].num_outputs = 1
            state.programs[HESSIAN].output_bytes[0] = hessian_size * cython.sizeof(
                ipnumber
            )

    def __dealloc__(self) -> None:
        state: cython.pointer[SolverState] = self.state
        if state == cython.NULL:
            return
        index: cython.int
        for index in range(INPUT_COUNT):
            destroy_buffer(state.inputs[index].buffer)
            free_aligned(state.inputs[index].host)
        free(state.jacobian_rows)
        free(state.jacobian_columns)
        free(state.hessian_rows)
        free(state.hessian_columns)
        free(state.cached_x)
        free(state.objective_value)
        free(state.gradient_values)
        free(state.constraint_values)
        free(state.jacobian_values)
        free(state.state_values)
        free(state)

    def notify(
        self,
        mode: cython.int,
        iteration: cython.int,
        objective: cython.double,
        primal_infeasibility: cython.double,
        dual_infeasibility: cython.double,
        barrier_parameter: cython.double,
        step_norm: cython.double,
        regularization_size: cython.double,
        dual_step_size: cython.double,
        primal_step_size: cython.double,
        line_search_trials: cython.int,
    ) -> cython.bint:
        """Calls the observer of the running solve and returns whether the solve goes on.

        An exception of the observer stops the solve and is raised again by `solve`.
        """
        try:
            variables = np.empty(self.state.variable_count)
            values: cython.double[::1] = variables
            GetIpoptCurrentIterate(
                self.state.problem,
                False,
                self.state.variable_count,
                cython.address(values[0]),
                cython.NULL,
                cython.NULL,
                self.state.constraint_count,
                cython.NULL,
                cython.NULL,
            )
            return bool(
                self.current_observer(
                    IterationState(
                        mode,
                        iteration,
                        objective,
                        primal_infeasibility,
                        dual_infeasibility,
                        barrier_parameter,
                        step_norm,
                        regularization_size,
                        dual_step_size,
                        primal_step_size,
                        line_search_trials,
                        variables,
                    )
                )
            )
        except BaseException as error:
            self.observer_error = error
            return False

    def solve(
        self,
        start: object,
        parameters: object,
        variable_lower: object,
        variable_upper: object,
        constraint_lower: object,
        constraint_upper: object,
        settings: Sequence[tuple[str, IpoptValue]] = (),
        warm_start: Multipliers | None = None,
        observer: Callable[[IterationState], bool] | None = None,
        initial_state: object = None,
    ) -> NativeResult:
        """Solves from `start` for the given parameters and bounds and returns the result.

        The bounds are flat vectors, one entry per variable or per constraint. `settings`
        are Ipopt options. The solve stops early when `observer` returns false. With
        `warm_start`, its multipliers are the starting multipliers, which Ipopt only reads
        when its option `warm_start_init_point` is `yes`. A problem with state starts from
        `initial_state`, which is required for it and must be left out otherwise.
        """
        with SOLVE_LOCK:
            return self.solve_locked(
                start,
                parameters,
                (variable_lower, variable_upper, constraint_lower, constraint_upper),
                settings,
                warm_start,
                observer,
                initial_state,
            )

    @cython.cfunc
    def solve_locked(
        self,
        start: object,
        parameters: object,
        bounds: tuple,
        settings: Sequence[tuple[str, IpoptValue]],
        warm_start: Multipliers | None,
        observer: Callable[[IterationState], bool] | None,
        initial_state: object,
    ) -> NativeResult:
        state: cython.pointer[SolverState] = self.state
        index: cython.int
        variable_count: cython.int = state.variable_count
        constraint_count: cython.int = state.constraint_count
        padded: cython.int = max(constraint_count, 1)
        parameter_size: cython.int = state.parameter_count
        state_size: cython.int = state.state_size
        if (initial_state is None) != (state_size == 0):
            raise ValueError(
                "A problem with state needs an initial state, and one without state "
                "takes none."
            )

        variables = float_array(start, variable_count, "starting point")
        parameter_values = float_array(parameters, parameter_size, "parameters")
        variable_lower = float_array(bounds[0], variable_count, "lower variable bounds")
        variable_upper = float_array(bounds[1], variable_count, "upper variable bounds")
        # Ipopt reads at least one entry of every bound array, so empty ones are padded.
        constraint_lower = np.zeros(padded)
        constraint_upper = np.zeros(padded)
        constraint_lower[:constraint_count] = float_array(
            bounds[2], constraint_count, "lower constraint bounds"
        )
        constraint_upper[:constraint_count] = float_array(
            bounds[3], constraint_count, "upper constraint bounds"
        )
        constraints = np.zeros(padded)
        constraint_multipliers = np.zeros(padded)
        lower_multipliers = np.zeros(variable_count)
        upper_multipliers = np.zeros(variable_count)
        if warm_start is not None:
            constraint_multipliers[:constraint_count] = float_array(
                warm_start.constraints, constraint_count, "constraint multipliers"
            )
            lower_multipliers[:] = float_array(
                warm_start.lower, variable_count, "lower multipliers"
            )
            upper_multipliers[:] = float_array(
                warm_start.upper, variable_count, "upper multipliers"
            )

        variable_view: cython.double[::1] = variables
        parameter_view: cython.double[::1] = parameter_values
        constraint_view: cython.double[::1] = constraints
        multiplier_view: cython.double[::1] = constraint_multipliers
        lower_view: cython.double[::1] = lower_multipliers
        upper_view: cython.double[::1] = upper_multipliers
        variable_lower_view: cython.double[::1] = variable_lower
        variable_upper_view: cython.double[::1] = variable_upper
        constraint_lower_view: cython.double[::1] = constraint_lower
        constraint_upper_view: cython.double[::1] = constraint_upper

        memcpy(
            state.inputs[PARAMETERS].host,
            cython.address(parameter_view[0]),
            parameter_size * cython.sizeof(ipnumber),
        )
        if state_size > 0:
            state_values = float_array(initial_state, state_size, "initial state")
            state_view: cython.double[::1] = state_values
            memcpy(
                cython.cast(cython.pointer[ipnumber], state.inputs[PARAMETERS].host)
                + parameter_size,
                cython.address(state_view[0]),
                state_size * cython.sizeof(ipnumber),
            )
        for index in range(CALLBACK_COUNT):
            state.callback_calls[index] = 0
            state.callback_hits[index] = 0
        for index in range(PROGRAM_COUNT):
            state.executions[index] = 0
            state.seconds[index] = 0.0
        state.unused_derivatives = 0
        state.iterations = 0
        # The parameters can differ from the last solve while the first point is the same,
        # so outputs from before would be wrong.
        state.cache_valid = 0
        state.cache_failed = 0
        state.derivatives_requested = 0

        problem: IpoptProblem = CreateIpoptProblem(
            variable_count,
            cython.address(variable_lower_view[0]),
            cython.address(variable_upper_view[0]),
            constraint_count,
            cython.address(constraint_lower_view[0]),
            cython.address(constraint_upper_view[0]),
            state.jacobian_size,
            state.hessian_size,
            0,
            evaluate_objective,
            evaluate_constraints,
            evaluate_gradient,
            evaluate_jacobian,
            evaluate_hessian,
        )
        if problem == cython.NULL:
            raise SolverError("Ipopt could not create the problem.")
        objective: ipnumber = 0.0
        status: cython.int
        self.observer_error = None
        try:
            # The banner is not part of the solve, so it is always suppressed.
            apply_setting(problem, "sb", "yes")
            for name, value in settings:
                apply_setting(problem, name, value)
            SetIntermediateCallback(problem, report_iteration)
            state.problem = problem
            self.current_observer = observer
            if observer is not None:
                state.observer = cython.cast(cython.p_void, self)
            with cython.nogil:
                status = IpoptSolve(
                    problem,
                    cython.address(variable_view[0]),
                    cython.address(constraint_view[0]),
                    cython.address(objective),
                    cython.address(multiplier_view[0]),
                    cython.address(lower_view[0]),
                    cython.address(upper_view[0]),
                    state,
                )
        finally:
            state.problem = cython.NULL
            state.observer = cython.NULL
            self.current_observer = None
            FreeIpoptProblem(problem)
        if self.observer_error is not None:
            error = self.observer_error
            self.observer_error = None
            raise error
        final_state = np.zeros(state_size)
        if state_size > 0:
            final_view: cython.double[::1] = final_state
            memcpy(
                cython.address(final_view[0]),
                cython.cast(cython.pointer[ipnumber], state.inputs[PARAMETERS].host)
                + parameter_size,
                state_size * cython.sizeof(ipnumber),
            )
        return NativeResult(
            status=status,
            objective=objective,
            variables=variables,
            constraints=constraints[:constraint_count],
            constraint_multipliers=constraint_multipliers[:constraint_count],
            lower_multipliers=lower_multipliers,
            upper_multipliers=upper_multipliers,
            iterations=state.iterations,
            callback_calls=tuple(
                [state.callback_calls[index] for index in range(CALLBACK_COUNT)]
            ),
            callback_hits=tuple(
                [state.callback_hits[index] for index in range(CALLBACK_COUNT)]
            ),
            executions=tuple(
                [state.executions[index] for index in range(PROGRAM_COUNT)]
            ),
            seconds=tuple([state.seconds[index] for index in range(PROGRAM_COUNT)]),
            unused_derivatives=state.unused_derivatives,
            state=final_state,
        )


@cython.cfunc
def copy_indices(indices: object) -> cython.pointer[cython.int]:
    """Returns a malloc'd copy of an index array, which the caller frees."""
    array = np.ascontiguousarray(indices, dtype=np.intc)
    view: cython.int[::1] = array
    size: cython.size_t = len(array)
    copy: cython.pointer[cython.int] = cython.cast(
        cython.pointer[cython.int], malloc(max(size, 1) * cython.sizeof(cython.int))
    )
    if copy == cython.NULL:
        raise MemoryError()
    if size > 0:
        memcpy(copy, cython.address(view[0]), size * cython.sizeof(cython.int))
    return copy


@cython.cfunc
def apply_setting(problem: IpoptProblem, name: str, value: object) -> cython.void:
    """Sets an option of a problem, dispatching on the type of the value."""
    keyword: bytes = name.encode()
    accepted: cython.bint
    match value:
        case bool():
            raise TypeError(
                f"The Ipopt option {name} must be a string, an integer or a float, "
                f"not {value!r}."
            )
        case str():
            text: bytes = value.encode()
            accepted = AddIpoptStrOption(problem, keyword, text)
        case int():
            accepted = AddIpoptIntOption(problem, keyword, value)
        case float():
            accepted = AddIpoptNumOption(problem, keyword, value)
        case _:
            raise TypeError(
                f"The Ipopt option {name} must be a string, an integer or a float, "
                f"not {value!r}."
            )
    if not accepted:
        raise ValueError(f"Ipopt rejected the option {name} with the value {value!r}.")
