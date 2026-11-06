"""Compiling the functions of a problem into programs for Ipopt's callbacks."""

import time
from collections.abc import Sequence
from typing import Any, NamedTuple, Never

import cyxla
import jax
import numpy as np
from cyxla import Executable, LoweredProgram
from jaxtyping import Float, PyTree

from xipopt import _cache, _programs
from xipopt._layout import (
    ParamsLayout,
    TreeLayout,
    params_layout,
    state_layout,
    variable_layout,
)
from xipopt._options import CompileOptions, Hessian
from xipopt._patterns import as_entries
from xipopt._problem_types import Problem, StatefulProblem
from xipopt._solver import NativeProblem
from xipopt._types import BuildStatistics, Program, Sense


class CompiledProblem(NamedTuple):
    """A problem whose functions are compiled, ready to solve."""

    problem: Problem | StatefulProblem | None
    """The problem that was compiled, or `None` for one that `load` returned."""
    options: CompileOptions
    sense: Sense
    variables: TreeLayout
    params: ParamsLayout
    constraints: TreeLayout
    """The structure and the shapes of the constraints that `Problem.evaluate` returns."""
    native: NativeProblem
    """The compiled programs with the buffers that Ipopt's callbacks share."""
    structure: _cache.Structure
    """Where the Jacobian and the Hessian have entries, in the order of their values."""
    statistics: BuildStatistics
    state: ParamsLayout | None = None
    """The layout of the state of a stateful problem, or `None` for a problem without state."""
    initial_state: Float[np.ndarray, " s"] = np.zeros(0)
    """The flat state that a solve starts from, empty for a problem without state."""

    def __reduce__(self) -> Never:
        """Refuses to pickle, since a compiled problem holds native buffers.

        `xipopt.dumps` and `xipopt.loads` move a compiled problem between processes. The
        trees of the variables and parameters are not part of the file, so unpickling could
        not return a problem that is ready to use.
        """
        raise TypeError(
            "A compiled problem cannot be pickled. Use xipopt.dumps to get bytes, and "
            "xipopt.loads with the variables and parameters to read them."
        )


def compile(  # noqa: A001
    problem: Problem | StatefulProblem,
    params: PyTree[Any] = None,
    options: CompileOptions = CompileOptions(),
) -> CompiledProblem:
    """Compiles the functions of a problem for parameters of the given structure and shapes.

    `params` may hold arrays or `jax.ShapeDtypeStruct`s, since only their structure,
    shapes and types matter. The result solves with any parameter values of that layout.
    The programs run in 64-bit floats, whatever the setting of `jax.enable_x64` is. A
    `StatefulProblem` compiles to programs that carry its state, which `solve` resets at the
    start of every solve.
    """
    cyxla.initialize(platform="cpu")
    with jax.enable_x64(True):
        return build(problem, params, options)


def build(
    problem: Problem | StatefulProblem, params: PyTree[Any], options: CompileOptions
) -> CompiledProblem:
    """Builds a problem in 64-bit mode, from the cache where it can."""
    start = time.perf_counter()
    variables = variable_layout(problem.variables)
    layout = params_layout(params)
    state = None if isinstance(problem, Problem) else state_layout(problem.state)
    functions = _programs.flat_functions(problem, variables, layout, state)
    constraint_size = functions.constraint_size
    multiplier_size = max(constraint_size, 1)
    arguments = _programs.abstract_arguments(
        variables, functions.parameter_size, multiplier_size
    )
    n = variables.size

    jacobian_given = (
        as_entries(problem.jacobian_pattern, (constraint_size, n))
        if problem.jacobian_pattern is not None and constraint_size > 0
        else None
    )
    hessian_given = (
        as_entries(problem.hessian_pattern, (n, n))
        if problem.hessian_pattern is not None and options.hessian is Hessian.EXACT
        else None
    )
    key = _cache.problem_key(
        _cache.jaxpr_digest(functions.evaluate, arguments.first_order),
        options,
        int(problem.sense),
        jacobian_given,
        hessian_given,
    )

    lowered: Sequence[LoweredProgram | None] | None = None
    structure = None
    detect_seconds = lower_seconds = 0.0
    lowered_count = 0
    disk_cache = options.disk_cache and not options.host_callbacks
    if disk_cache:
        lowered = cyxla.load_lowered(key)
        structure = _cache.load_structure(key)
    if lowered is None or structure is None or len(lowered) != len(Program):
        step = time.perf_counter()
        derivatives = _programs.build_derivatives(
            problem, functions, variables, multiplier_size, options
        )
        detect_seconds = time.perf_counter() - step
        structure = structure_of(derivatives)
        step = time.perf_counter()
        lowered = lower_programs(functions, derivatives, arguments, options)
        lower_seconds = time.perf_counter() - step
        lowered_count = sum(program is not None for program in lowered)
        if disk_cache:
            cyxla.store_lowered(key, lowered)
            _cache.store_structure(key, structure)

    if constraint_size > 0 and structure.jacobian_rows.size == 0:
        raise ValueError(
            "The constraints do not depend on the variables, so their Jacobian has no "
            "entries, which Ipopt does not accept."
        )
    executables, compile_statistics = compile_programs(lowered, disk_cache)
    check_output_sizes(executables, n, constraint_size, structure, functions.state_size)
    native = make_native(
        executables,
        variables,
        layout,
        functions.constraints,
        structure,
        options,
        functions.state_size,
    )
    return CompiledProblem(
        problem=problem,
        options=options,
        sense=problem.sense,
        variables=variables,
        params=layout,
        constraints=functions.constraints,
        native=native,
        structure=structure,
        statistics=_cache.combine(
            compile_statistics,
            detect_seconds,
            lower_seconds,
            lowered_count,
            time.perf_counter() - start,
        ),
        state=state,
        initial_state=(
            np.zeros(0)
            if state is None or isinstance(problem, Problem)
            else state.flatten(problem.state)
        ),
    )


def make_native(
    executables: Sequence[Executable | None],
    variables: TreeLayout,
    layout: ParamsLayout,
    constraints: TreeLayout,
    structure: _cache.Structure,
    options: CompileOptions,
    state_size: int = 0,
) -> NativeProblem:
    """Returns the native problem for compiled programs and a sparsity structure.

    `layout` describes the parameters, and `state_size` is the length of the state that
    follows them in the parameter input of the programs.
    """
    return NativeProblem(
        executables,
        variables.size,
        constraints.size,
        layout.size,
        structure.jacobian_rows,
        structure.jacobian_columns,
        structure.hessian_rows,
        structure.hessian_columns,
        options.zero_copy,
        state_size,
    )


def structure_of(derivatives: _programs.DerivativeFunctions) -> _cache.Structure:
    """Returns the sparsity structure of the derivatives."""
    empty = np.zeros(0, dtype=np.intc)
    hessian = derivatives.hessian
    return _cache.Structure(
        derivatives.first_order.rows,
        derivatives.first_order.columns,
        empty if hessian is None else hessian.rows,
        empty if hessian is None else hessian.columns,
    )


def lower_programs(
    functions: _programs.FlatFunctions,
    derivatives: _programs.DerivativeFunctions,
    arguments: _programs.ProgramArguments,
    options: CompileOptions,
) -> tuple[LoweredProgram | None, ...]:
    """Lowers the two programs of a problem, in the order of `Program`.

    A Hessian that is not compiled, for a limited-memory Hessian or one without entries, is
    `None`.
    """

    def lower(function: _programs.FlatFunction | None, shapes: Any) -> Any:
        if function is None:
            return None
        try:
            return cyxla.lower(
                function,
                shapes,
                platform="cpu",
                custom_calls=options.custom_calls,
                debug_info=options.debug_info,
                host_callbacks=options.host_callbacks,
            )
        except ValueError as error:
            if "no compatibility guarantees" not in str(error):
                raise
            raise ValueError(
                "A program calls custom call targets that JAX has no compatibility "
                "guarantees for, such as the targets of a third-party library. Allow "
                'them with `CompileOptions(custom_calls="detect")`, or name the targets '
                "in `custom_calls`."
            ) from error

    hessian = derivatives.hessian
    return (
        lower(derivatives.first_order.function, arguments.first_order),
        lower(
            None if hessian is None or hessian.rows.size == 0 else hessian.function,
            arguments.second_order,
        ),
    )


def compile_programs(
    lowered: Sequence[LoweredProgram | None], disk_cache: bool
) -> tuple[tuple[Executable | None, ...], cyxla.CompileStatistics]:
    """Compiles lowered programs, or loads their executables from the disk cache if allowed."""
    statistics = cyxla.CompileStatistics()
    executables: list[Executable | None] = []
    for program in lowered:
        if program is None:
            executables.append(None)
            continue
        executable, compiled = cyxla.compile_program(program, disk_cache=disk_cache)
        statistics = statistics.add(compiled)
        executables.append(executable)
    return tuple(executables), statistics


def check_output_sizes(
    executables: Sequence[Executable | None],
    variable_count: int,
    constraint_size: int,
    structure: _cache.Structure,
    state_size: int = 0,
) -> None:
    """Checks that each program writes as many values as Ipopt expects."""
    expected = {
        Program.FIRST_ORDER: (
            1,
            variable_count,
            constraint_size,
            structure.jacobian_rows.size,
            *((state_size,) if state_size > 0 else ()),
        ),
        Program.HESSIAN: (structure.hessian_rows.size,),
    }
    for program, executable in zip(Program, executables, strict=True):
        if executable is None:
            continue
        sizes = tuple(8 * size for size in expected[program])
        if executable.output_bytes != sizes:
            raise RuntimeError(
                f"The {program.name.lower()} program writes {executable.output_bytes} "
                f"bytes, but {sizes} are expected."
            )
