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
from cython.cimports.cyxla._pjrt import PJRT_Buffer_Type_F64
from cython.cimports.cyxla._runtime import (
    ABI_VERSION,
    MAX_OUTPUTS,
    Executable,
    alias_buffer,
    aligned_memory,
    destroy_buffer,
    execute,
    free_aligned,
    now,
)
from cython.cimports.libc.stdlib import calloc, free, malloc
from cython.cimports.libc.string import memcpy
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
    CONSTRAINTS,
    GRADIENT,
    HESSIAN,
    INPUT_COUNT,
    JACOBIAN,
    MULTIPLIERS,
    OBJECTIVE,
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
    calls: tuple[int, ...]
    """Executions of each program, in the order of `PROGRAM_NAMES`."""
    seconds: tuple[float, ...]
    """Seconds spent executing each program."""


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def run_program(
    state: cython.pointer[SolverState], program: cython.int, destination: cython.p_void
) -> ipbool:
    """Executes a program into `destination` and returns whether it succeeded.

    A program that was not compiled is one that has nothing to compute, such as the
    constraints of a problem without any, and counts as success.
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
    state.calls[program] += 1
    state.seconds[program] += now() - start
    return status == 0


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
    memcpy(state.inputs[VARIABLES].host, x, n * cython.sizeof(ipnumber))
    return run_program(state, OBJECTIVE, objective)


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
    memcpy(state.inputs[VARIABLES].host, x, n * cython.sizeof(ipnumber))
    return run_program(state, GRADIENT, gradient)


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
    memcpy(state.inputs[VARIABLES].host, x, n * cython.sizeof(ipnumber))
    return run_program(state, CONSTRAINTS, constraints)


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
    memcpy(state.inputs[VARIABLES].host, x, n * cython.sizeof(ipnumber))
    return run_program(state, JACOBIAN, values)


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
        found = library_path(cython.cast(cython.size_t, address))  # pragma: no cover
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

    `executables` holds the programs in the order objective, gradient, constraints,
    Jacobian and Hessian of the Lagrangian, where `None` leaves a program out. Programs read
    the variables and the parameters, and the Hessian program also the objective factor and
    the constraint multipliers. The Jacobian and Hessian entries follow the given row and
    column indices, and the Hessian holds the lower triangle only.
    """

    state: cython.pointer[SolverState]
    executables: tuple
    """The executables, which have to outlive the handles that the state holds."""
    variable_lower: object
    variable_upper: object
    constraint_lower: object
    constraint_upper: object
    current_observer: object
    observer_error: object

    def __cinit__(
        self,
        executables: Sequence[Executable | None],
        variable_lower: object,
        variable_upper: object,
        constraint_lower: object,
        constraint_upper: object,
        parameter_size: cython.int,
        jacobian_rows: object,
        jacobian_columns: object,
        hessian_rows: object,
        hessian_columns: object,
        zero_copy: cython.bint = True,
    ) -> None:
        self.state = cython.NULL
        self.executables = tuple(executables)
        self.current_observer = None
        self.observer_error = None
        if len(self.executables) != PROGRAM_COUNT:
            raise ValueError(f"Expected {PROGRAM_COUNT} programs.")
        variable_count: cython.int = len(variable_lower)
        constraint_count: cython.int = len(constraint_lower)
        self.variable_lower = float_array(
            variable_lower, variable_count, "lower bounds"
        )
        self.variable_upper = float_array(
            variable_upper, variable_count, "upper bounds"
        )
        # Ipopt reads at least one entry of every bound array, so empty ones are padded.
        padded: cython.int = max(constraint_count, 1)
        self.constraint_lower = np.zeros(padded)
        self.constraint_upper = np.zeros(padded)
        self.constraint_lower[:constraint_count] = constraint_lower
        self.constraint_upper[:constraint_count] = constraint_upper

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
        state.jacobian_rows = copy_indices(jacobian_rows)
        state.jacobian_columns = copy_indices(jacobian_columns)
        state.hessian_rows = copy_indices(hessian_rows)
        state.hessian_columns = copy_indices(hessian_columns)

        sizes: cython.int[INPUT_COUNT]
        sizes[VARIABLES] = variable_count
        sizes[PARAMETERS] = max(parameter_size, 1)
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

        outputs: cython.int[PROGRAM_COUNT]
        outputs[OBJECTIVE] = 1
        outputs[GRADIENT] = variable_count
        outputs[CONSTRAINTS] = constraint_count
        outputs[JACOBIAN] = jacobian_size
        outputs[HESSIAN] = hessian_size
        executable: Executable
        for index in range(PROGRAM_COUNT):
            if self.executables[index] is None:
                continue
            executable = self.executables[index]
            state.programs[index].handle = executable.handle
            state.programs[index].num_inputs = (
                INPUT_COUNT if index == HESSIAN else OBJECTIVE_FACTOR
            )
            state.programs[index].num_outputs = 1
            state.programs[index].output_bytes[0] = outputs[index] * cython.sizeof(
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
        settings: Sequence[tuple[str, IpoptValue]] = (),
        warm_start: Multipliers | None = None,
        observer: Callable[[IterationState], bool] | None = None,
    ) -> NativeResult:
        """Solves from `start` for the given parameters and returns the result.

        `settings` are Ipopt options. The solve stops early when `observer` returns false.
        With `warm_start`, its multipliers are the starting multipliers, which Ipopt only
        reads when its option `warm_start_init_point` is `yes`.
        """
        with SOLVE_LOCK:
            return self.solve_locked(start, parameters, settings, warm_start, observer)

    @cython.cfunc
    def solve_locked(
        self,
        start: object,
        parameters: object,
        settings: Sequence[tuple[str, IpoptValue]],
        warm_start: Multipliers | None,
        observer: Callable[[IterationState], bool] | None,
    ) -> NativeResult:
        state: cython.pointer[SolverState] = self.state
        index: cython.int
        variable_count: cython.int = state.variable_count
        constraint_count: cython.int = state.constraint_count
        padded: cython.int = max(constraint_count, 1)
        parameter_size: cython.int = cython.cast(
            cython.int, state.inputs[PARAMETERS].size
        )

        variables = float_array(start, variable_count, "starting point")
        parameter_values = float_array(parameters, parameter_size, "parameters")
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
        variable_lower_view: cython.double[::1] = self.variable_lower
        variable_upper_view: cython.double[::1] = self.variable_upper
        constraint_lower_view: cython.double[::1] = self.constraint_lower
        constraint_upper_view: cython.double[::1] = self.constraint_upper

        memcpy(
            state.inputs[PARAMETERS].host,
            cython.address(parameter_view[0]),
            parameter_size * cython.sizeof(ipnumber),
        )
        for index in range(PROGRAM_COUNT):
            state.calls[index] = 0
            state.seconds[index] = 0.0
        state.iterations = 0

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
        return NativeResult(
            status=status,
            objective=objective,
            variables=variables,
            constraints=constraints[:constraint_count],
            constraint_multipliers=constraint_multipliers[:constraint_count],
            lower_multipliers=lower_multipliers,
            upper_multipliers=upper_multipliers,
            iterations=state.iterations,
            calls=tuple([state.calls[index] for index in range(PROGRAM_COUNT)]),
            seconds=tuple([state.seconds[index] for index in range(PROGRAM_COUNT)]),
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
