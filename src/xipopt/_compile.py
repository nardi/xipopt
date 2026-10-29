"""Compiling the functions of a problem into programs for Ipopt's callbacks."""

import time
from collections.abc import Sequence
from typing import Any, NamedTuple

import cyxla
import jax
import numpy as np
from cyxla import Executable, LoweredProgram
from jaxtyping import Float, PyTree

from xipopt import _cache, _programs
from xipopt._layout import (
    INFINITY,
    ParamsLayout,
    VariableLayout,
    params_layout,
    variable_layout,
)
from xipopt._options import CompileOptions, Hessian
from xipopt._patterns import as_entries
from xipopt._problem_types import Problem
from xipopt._solver import NativeProblem
from xipopt._types import BuildStatistics, Program


class CompiledProblem(NamedTuple):
    """A problem whose functions are compiled, ready to solve."""

    problem: Problem
    options: CompileOptions
    variables: VariableLayout
    params: ParamsLayout
    constraint_shapes: tuple[tuple[int, ...], ...]
    """The shape of the output of each constraint."""
    native: NativeProblem
    """The compiled programs with the buffers that Ipopt's callbacks share."""
    structure: _cache.Structure
    """Where the Jacobian and the Hessian have entries, in the order of their values."""
    statistics: BuildStatistics

    @property
    def constraint_size(self) -> int:
        """Number of scalar constraints."""
        return sum(int(np.prod(shape)) for shape in self.constraint_shapes)


def compile(  # noqa: A001
    problem: Problem,
    params: PyTree[Any] = None,
    options: CompileOptions = CompileOptions(),
) -> CompiledProblem:
    """Compiles the functions of a problem for parameters of the given structure and shapes.

    `params` may hold arrays or `jax.ShapeDtypeStruct`s, since only their structure,
    shapes and types matter. The result solves with any parameter values of that layout.
    The programs run in 64-bit floats, whatever the setting of `jax.enable_x64` is.
    """
    cyxla.initialize(platform="cpu")
    with jax.enable_x64(True):
        return build(problem, params, options)


def build(
    problem: Problem, params: PyTree[Any], options: CompileOptions
) -> CompiledProblem:
    """Builds a problem in 64-bit mode, from the cache where it can."""
    start = time.perf_counter()
    variables = variable_layout(problem.variables)
    layout = params_layout(params)
    functions = _programs.flat_functions(problem, variables, layout)
    constraint_size = functions.constraint_size
    multiplier_size = max(constraint_size, 1)
    arguments = _programs.abstract_arguments(variables, layout, multiplier_size)
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
        _cache.jaxpr_digest(functions.objective, arguments.first_order),
        b""
        if functions.constraints is None
        else _cache.jaxpr_digest(functions.constraints, arguments.first_order),
        options,
        int(problem.sense),
        jacobian_given,
        hessian_given,
    )

    lowered: Sequence[LoweredProgram | None] | None = None
    structure = None
    detect_seconds = lower_seconds = 0.0
    lowered_count = 0
    if options.disk_cache:
        lowered = cyxla.load_lowered(key)
        structure = _cache.load_structure(key)
    if lowered is None or structure is None or len(lowered) != len(Program):
        step = time.perf_counter()
        derivatives = _programs.build_derivatives(
            problem, functions, variables, layout, multiplier_size, options
        )
        detect_seconds = time.perf_counter() - step
        structure = structure_of(derivatives)
        step = time.perf_counter()
        lowered = lower_programs(functions, derivatives, arguments)
        lower_seconds = time.perf_counter() - step
        lowered_count = sum(program is not None for program in lowered)
        if options.disk_cache:
            cyxla.store_lowered(key, lowered)
            _cache.store_structure(key, structure)

    if constraint_size > 0 and structure.jacobian_rows.size == 0:
        raise ValueError(
            "The constraints do not depend on the variables, so their Jacobian has no "
            "entries, which Ipopt does not accept."
        )
    executables, compile_statistics = compile_programs(lowered, options)
    check_output_sizes(executables, n, constraint_size, structure)
    lower_bound, upper_bound = constraint_bounds(problem, functions.constraint_shapes)
    native = NativeProblem(
        executables,
        variables.lower,
        variables.upper,
        lower_bound,
        upper_bound,
        layout.size,
        structure.jacobian_rows,
        structure.jacobian_columns,
        structure.hessian_rows,
        structure.hessian_columns,
        options.zero_copy,
    )
    return CompiledProblem(
        problem=problem,
        options=options,
        variables=variables,
        params=layout,
        constraint_shapes=functions.constraint_shapes,
        native=native,
        structure=structure,
        statistics=_cache.combine(
            compile_statistics,
            detect_seconds,
            lower_seconds,
            lowered_count,
            time.perf_counter() - start,
        ),
    )


def structure_of(derivatives: _programs.DerivativeFunctions) -> _cache.Structure:
    """Returns the sparsity structure of the derivatives."""
    empty = np.zeros(0, dtype=np.intc)
    jacobian = derivatives.jacobian
    hessian = derivatives.hessian
    return _cache.Structure(
        empty if jacobian is None else jacobian.rows,
        empty if jacobian is None else jacobian.columns,
        empty if hessian is None else hessian.rows,
        empty if hessian is None else hessian.columns,
    )


def lower_programs(
    functions: _programs.FlatFunctions,
    derivatives: _programs.DerivativeFunctions,
    arguments: _programs.ProgramArguments,
) -> tuple[LoweredProgram | None, ...]:
    """Lowers the five programs of a problem, in the order of `Program`.

    A program with nothing to compute, such as the Jacobian of a problem without
    constraints, is `None`.
    """

    def lower(function: _programs.FlatFunction | None, shapes: Any) -> Any:
        if function is None:
            return None
        return cyxla.lower(function, shapes, platform="cpu")

    jacobian = derivatives.jacobian
    hessian = derivatives.hessian
    return (
        lower(functions.objective, arguments.first_order),
        lower(derivatives.gradient, arguments.first_order),
        lower(functions.constraints, arguments.first_order),
        lower(
            None if jacobian is None or jacobian.rows.size == 0 else jacobian.function,
            arguments.first_order,
        ),
        lower(
            None if hessian is None or hessian.rows.size == 0 else hessian.function,
            arguments.second_order,
        ),
    )


def compile_programs(
    lowered: Sequence[LoweredProgram | None], options: CompileOptions
) -> tuple[tuple[Executable | None, ...], cyxla.CompileStatistics]:
    """Compiles lowered programs, or loads their executables from the disk cache."""
    statistics = cyxla.CompileStatistics()
    executables: list[Executable | None] = []
    for program in lowered:
        if program is None:
            executables.append(None)
            continue
        executable, compiled = cyxla.compile_program(
            program, disk_cache=options.disk_cache
        )
        statistics = statistics.add(compiled)
        executables.append(executable)
    return tuple(executables), statistics


def check_output_sizes(
    executables: Sequence[Executable | None],
    variable_count: int,
    constraint_size: int,
    structure: _cache.Structure,
) -> None:
    """Checks that each program writes as many values as Ipopt expects."""
    expected = {
        Program.OBJECTIVE: 1,
        Program.GRADIENT: variable_count,
        Program.CONSTRAINTS: constraint_size,
        Program.JACOBIAN: structure.jacobian_rows.size,
        Program.HESSIAN: structure.hessian_rows.size,
    }
    for program, executable in zip(Program, executables, strict=True):
        if executable is None:
            continue
        if executable.output_bytes != (8 * expected[program],):
            raise RuntimeError(
                f"The {program.name.lower()} program writes {executable.output_bytes} "
                f"bytes, but {8 * expected[program]} are expected."
            )


def constraint_bounds(
    problem: Problem, shapes: tuple[tuple[int, ...], ...]
) -> tuple[Float[np.ndarray, " m"], Float[np.ndarray, " m"]]:
    """Returns the lower and upper bounds of all constraints as flat vectors."""
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    for constraint, shape in zip(problem.constraints, shapes, strict=True):
        try:
            lower = np.broadcast_to(np.asarray(constraint.lower, np.float64), shape)
            upper = np.broadcast_to(np.asarray(constraint.upper, np.float64), shape)
        except ValueError as error:
            raise ValueError(
                f"The bounds of a constraint do not broadcast to its shape {shape}."
            ) from error
        if np.any(lower > upper):
            raise ValueError("A constraint has a lower bound above its upper bound.")
        lowers.append(np.clip(lower.ravel(), -INFINITY, INFINITY))
        uppers.append(np.clip(upper.ravel(), -INFINITY, INFINITY))
    if not lowers:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(lowers), np.concatenate(uppers)
