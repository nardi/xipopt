"""The five functions that Ipopt's callbacks evaluate, as flat JAX functions.

Every function takes the flat vector of variables `x` and the flat vector of parameters `p`.
The Hessian of the Lagrangian also takes the factor of the objective and the multipliers
of the constraints. Derivatives are evaluated only at the entries of a sparsity pattern, in
the order of the rows and columns that come with them.
"""

import importlib.util
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

from xipopt._layout import ParamsLayout, VariableLayout
from xipopt._options import CompileOptions, Derivatives, Hessian
from xipopt._patterns import (
    Entries,
    as_entries,
    dense_entries,
    lower_triangle,
    symmetric_entries,
)
from xipopt._problem_types import Constraint, Problem
from xipopt._types import Sense

FlatFunction = Callable[..., Any]
"""A function of flat float64 vectors that returns a float64 vector or scalar.

It is typed loosely because JAX traces it with arrays and shape structures alike.
"""


class SparsityDetectionError(RuntimeError):
    """The sparsity pattern of a derivative could not be detected."""


class FlatFunctions(NamedTuple):
    """The functions of a problem on flat vectors."""

    objective: FlatFunction
    """Returns the scalar objective, negated for a problem that maximizes."""
    constraints: FlatFunction | None
    """Returns all constraint values as one vector, or `None` without constraints."""
    constraint_shapes: tuple[tuple[int, ...], ...]
    """The shape of the output of each constraint."""
    lagrangian: FlatFunction
    """Returns `factor * objective + multipliers @ constraints` for `(x, p, factor, multipliers)`."""

    @property
    def constraint_size(self) -> int:
        """Number of scalar constraints."""
        return sum(int(np.prod(shape)) for shape in self.constraint_shapes)


class Derivative(NamedTuple):
    """A derivative evaluated at the entries of a pattern."""

    function: FlatFunction
    """Returns the values of the entries, in the order of `rows` and `columns`."""
    rows: Int[np.ndarray, " nnz"]
    columns: Int[np.ndarray, " nnz"]


class DerivativeFunctions(NamedTuple):
    """The derivatives that Ipopt asks for."""

    gradient: FlatFunction
    jacobian: Derivative | None
    hessian: Derivative | None


def asdex_installed() -> bool:
    """Returns whether `asdex` can be imported."""
    return importlib.util.find_spec("asdex") is not None


def flat_functions(
    problem: Problem,
    variables: VariableLayout,
    params: ParamsLayout,
) -> FlatFunctions:
    """Returns the functions of a problem on flat vectors."""
    sign = 1.0 if problem.sense is Sense.MINIMIZE else -1.0
    x_aval = jax.ShapeDtypeStruct((variables.size,), jnp.float64)
    p_aval = jax.ShapeDtypeStruct((params.size,), jnp.float64)

    def traced_arguments(x: Any, p: Any) -> tuple[Any, Any]:
        return variables.unflatten_traced(x), params.unflatten_traced(p)

    if problem.objective is None:

        def objective(x: Any, p: Any) -> Any:
            # A feasibility problem has the constant objective zero, which still depends on `x`
            # so that its derivatives have the right shapes.
            return 0.0 * jnp.sum(x)
    else:
        user_objective = problem.objective

        def objective(x: Any, p: Any) -> Any:
            value = jnp.asarray(user_objective(*traced_arguments(x, p)))
            return sign * value.astype(jnp.float64)

        shape = jax.eval_shape(objective, x_aval, p_aval).shape
        if shape != ():
            raise ValueError(f"The objective must return a scalar, not shape {shape}.")

    constraints_list = list(problem.constraints)
    for constraint in constraints_list:
        if not isinstance(constraint, Constraint):
            raise TypeError(
                f"A constraint must be a Constraint, not {type(constraint).__name__}."
            )

    def constraint_outputs(x: Any, p: Any) -> list[Any]:
        arguments = traced_arguments(x, p)
        return [
            jnp.ravel(jnp.asarray(constraint.function(*arguments)).astype(jnp.float64))
            for constraint in constraints_list
        ]

    constraints: FlatFunction | None = None
    shapes: tuple[tuple[int, ...], ...] = ()
    if constraints_list:
        shapes = tuple(
            jax.eval_shape(
                lambda x, p, constraint=constraint: jnp.asarray(
                    constraint.function(*traced_arguments(x, p))
                ),
                x_aval,
                p_aval,
            ).shape
            for constraint in constraints_list
        )

        def constraints(x: Any, p: Any) -> Any:
            return jnp.concatenate(constraint_outputs(x, p))

    def lagrangian(x: Any, p: Any, factor: Any, multipliers: Any) -> Any:
        value = factor * objective(x, p)
        if constraints is not None:
            count = sum(int(np.prod(shape)) for shape in shapes)
            value = value + jnp.dot(multipliers[:count], constraints(x, p))
        return value

    return FlatFunctions(objective, constraints, shapes, lagrangian)


def sample_arguments(
    variables: VariableLayout, params: ParamsLayout, multiplier_size: int
) -> tuple[Any, ...]:
    """Returns concrete arguments of the Lagrangian, whose values detection ignores."""
    return (
        jnp.zeros(variables.size),
        jnp.zeros(params.size),
        jnp.asarray(1.0),
        jnp.zeros(multiplier_size),
    )


def load_asdex(options: CompileOptions) -> Any | None:
    """Returns the `asdex` module when the options call for it, or `None` for dense derivatives.

    The return type is `Any` because `asdex` is an optional dependency without type
    information that this package can name.
    """
    match options.derivatives:
        case Derivatives.DENSE:
            return None
        case Derivatives.SPARSE:
            import asdex

            return asdex
        case _:
            # Automatic selection, which uses asdex when it can.
            if asdex_installed():
                import asdex

                return asdex
            warnings.warn(
                "asdex is not installed, so derivatives are dense. Install asdex to "
                "detect sparsity, or give the patterns of the problem.",
                stacklevel=3,
            )
            return None


def build_derivatives(
    problem: Problem,
    functions: FlatFunctions,
    variables: VariableLayout,
    params: ParamsLayout,
    multiplier_size: int,
    options: CompileOptions,
) -> DerivativeFunctions:
    """Returns the derivatives of a problem with their patterns."""
    asdex = load_asdex(options)
    samples = sample_arguments(variables, params, multiplier_size)
    n = variables.size
    m = functions.constraint_size

    jacobian = None
    if functions.constraints is not None:
        given = (
            None
            if problem.jacobian_pattern is None
            else as_entries(problem.jacobian_pattern, (m, n))
        )
        jacobian = jacobian_derivative(
            functions.constraints, given, samples[:2], (m, n), asdex
        )

    hessian = None
    if options.hessian is Hessian.EXACT:
        given = (
            None
            if problem.hessian_pattern is None
            else as_entries(problem.hessian_pattern, (n, n))
        )
        hessian = hessian_derivative(
            functions.lagrangian, given, samples, (n, n), asdex
        )
    return DerivativeFunctions(jax.grad(functions.objective), jacobian, hessian)


def jacobian_derivative(
    function: FlatFunction,
    given: Entries | None,
    samples: tuple[Any, ...],
    shape: tuple[int, int],
    asdex: Any | None,
) -> Derivative:
    """Returns the Jacobian of a function of `(x, p)` at the entries of its pattern."""
    if asdex is None:
        entries = given if given is not None else dense_entries(shape)
        rows, columns = jnp.asarray(entries.rows), jnp.asarray(entries.columns)
        differentiate = jax.jacfwd if shape[1] <= shape[0] else jax.jacrev

        def dense(x: Any, p: Any) -> Any:
            return differentiate(function)(x, p)[rows, columns]

        return Derivative(dense, entries.rows, entries.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        coloring = colored_pattern(
            asdex, "jacobian", function, given, samples, shape, "the Jacobian"
        )
    evaluate = asdex.jacobian_from_coloring(function, coloring)
    entries = entries_of(coloring.sparsity, shape)

    def sparse(x: Any, p: Any) -> Any:
        return evaluate(x, p).data

    return Derivative(sparse, entries.rows, entries.columns)


def hessian_derivative(
    function: FlatFunction,
    given: Entries | None,
    samples: tuple[Any, ...],
    shape: tuple[int, int],
    asdex: Any | None,
) -> Derivative:
    """Returns the lower triangle of the Hessian of a function of `(x, p, factor, multipliers)`."""
    if asdex is None:
        entries = lower_triangle(given if given is not None else dense_entries(shape))
        rows, columns = jnp.asarray(entries.rows), jnp.asarray(entries.columns)

        def dense(x: Any, p: Any, factor: Any, multipliers: Any) -> Any:
            matrix = jax.hessian(function)(x, p, factor, multipliers)
            return matrix[rows, columns]

        return Derivative(dense, entries.rows, entries.columns)

    symmetric = None if given is None else symmetric_entries(given)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        coloring = colored_pattern(
            asdex, "hessian", function, symmetric, samples, shape, "the Hessian"
        )
    evaluate = asdex.hessian_from_coloring(function, coloring)
    full = entries_of(coloring.sparsity, shape)
    # Ipopt wants the lower triangle only, so the program keeps those entries of the symmetric
    # matrix that asdex evaluates.
    keep = np.flatnonzero(full.rows >= full.columns)

    def sparse(x: Any, p: Any, factor: Any, multipliers: Any) -> Any:
        return evaluate(x, p, factor, multipliers).data[keep]

    return Derivative(sparse, full.rows[keep], full.columns[keep])


def entries_of(pattern: Any, shape: tuple[int, int]) -> Entries:
    """Returns the entries of an `asdex.SparsityPattern` in the order `asdex` evaluates them.

    The entries are not sorted again, because the compressed values follow this order.
    """
    return Entries(
        np.asarray(pattern.rows).astype(np.intc),
        np.asarray(pattern.cols).astype(np.intc),
        shape,
    )


def colored_pattern(
    asdex: Any,
    kind: str,
    function: FlatFunction,
    given: Entries | None,
    samples: tuple[Any, ...],
    shape: tuple[int, int],
    description: str,
) -> Any:
    """Returns the coloring of a derivative, from the given pattern or by detecting it.

    A given pattern has to carry the shapes of the arguments, which `asdex` needs to evaluate
    the compressed derivative.
    """
    if given is None:
        try:
            if kind == "jacobian":
                return asdex.jacobian_coloring(function, *samples)
            return asdex.hessian_coloring(function, *samples)
        except Exception as error:
            raise SparsityDetectionError(
                f"The sparsity of {description} could not be detected: {error}\n"
                "Give the pattern in the Problem, set `derivatives=Derivatives.DENSE` "
                "in the CompileOptions, or use `hessian=Hessian.LIMITED_MEMORY`."
            ) from error
    avals = tuple(
        jax.ShapeDtypeStruct(np.shape(sample), jnp.float64) for sample in samples
    )
    pattern = asdex.SparsityPattern.from_coo(
        given.rows, given.columns, shape, input_avals=avals
    )
    if kind == "jacobian":
        return asdex.jacobian_coloring_from_sparsity(pattern)
    return asdex.hessian_coloring_from_sparsity(pattern)


class ProgramArguments(NamedTuple):
    """The abstract arguments of the programs, which are float64 vectors and a scalar."""

    first_order: tuple[jax.ShapeDtypeStruct, ...]
    """The variables and the parameters, which the objective, constraints and their first
    derivatives take."""
    second_order: tuple[jax.ShapeDtypeStruct, ...]
    """The arguments of the Hessian of the Lagrangian, which also has the objective factor
    and the multipliers."""


def abstract_arguments(
    variables: VariableLayout, params: ParamsLayout, multiplier_size: int
) -> ProgramArguments:
    """Returns the abstract arguments of the programs."""
    x = jax.ShapeDtypeStruct((variables.size,), jnp.float64)
    p = jax.ShapeDtypeStruct((params.size,), jnp.float64)
    factor = jax.ShapeDtypeStruct((), jnp.float64)
    multipliers = jax.ShapeDtypeStruct((multiplier_size,), jnp.float64)
    return ProgramArguments((x, p), (x, p, factor, multipliers))
