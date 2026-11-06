"""The two functions that Ipopt's callbacks evaluate, as flat JAX functions.

Every function takes the flat vector of variables `x` and the flat vector of parameters `p`.
The first-order function returns the objective, its gradient, the constraint values and the
values of the Jacobian in one pass, and the Hessian of the Lagrangian also takes the factor of
the objective and the multipliers of the constraints. Derivatives are evaluated only at the
entries of a sparsity pattern, in the order of the rows and columns that come with them.
"""

import importlib.util
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

from xipopt._layout import ParamsLayout, TreeLayout, tree_layout
from xipopt._options import CompileOptions, Derivatives, Hessian
from xipopt._patterns import (
    Entries,
    as_entries,
    dense_entries,
    lower_triangle,
    symmetric_entries,
)
from xipopt._problem_types import Evaluation, Problem
from xipopt._types import Sense

FlatFunction = Callable[..., Any]
"""A function of flat float64 vectors that returns a float64 vector or scalar.

It is typed loosely because JAX traces it with arrays and shape structures alike.
"""


class SparsityDetectionError(RuntimeError):
    """The sparsity pattern of a derivative could not be detected."""


class FlatFunctions(NamedTuple):
    """The functions of a problem on flat vectors."""

    evaluate: FlatFunction
    """Returns the objective, negated for a problem that maximizes, followed by all constraint
    values: one vector of length `1 + m`."""
    constraints: TreeLayout
    """The structure and the shapes of the arrays in `Evaluation.constraints`."""
    lagrangian: FlatFunction
    """Returns `factor * objective + multipliers @ constraints` for `(x, p, factor, multipliers)`."""

    @property
    def constraint_size(self) -> int:
        """Number of scalar constraints."""
        return self.constraints.size


class Derivative(NamedTuple):
    """A derivative evaluated at the entries of a pattern."""

    function: FlatFunction
    """Returns the values of the entries, in the order of `rows` and `columns`."""
    rows: Int[np.ndarray, " nnz"]
    columns: Int[np.ndarray, " nnz"]


class DerivativeFunctions(NamedTuple):
    """The programs that the callbacks of Ipopt run."""

    first_order: Derivative
    """Returns `(objective, gradient, constraints, jacobian_values)` for `(x, p)`. The rows
    and columns are those of the Jacobian entries, with the rows counted over the constraints."""
    hessian: Derivative | None


def asdex_installed() -> bool:
    """Returns whether `asdex` can be imported."""
    return importlib.util.find_spec("asdex") is not None


def flat_functions(
    problem: Problem,
    variables: TreeLayout,
    params: ParamsLayout,
) -> FlatFunctions:
    """Returns the functions of a problem on flat vectors."""
    sign = 1.0 if problem.sense is Sense.MINIMIZE else -1.0
    x_aval = jax.ShapeDtypeStruct((variables.size,), jnp.float64)
    p_aval = jax.ShapeDtypeStruct((params.size,), jnp.float64)

    def evaluation_at(x: Any, p: Any) -> Evaluation:
        result = problem.evaluate(
            variables.unflatten_traced(x), params.unflatten_traced(p)
        )
        if not isinstance(result, Evaluation):
            raise TypeError(
                "`Problem.evaluate` must return an Evaluation, not "
                f"{type(result).__name__}."
            )
        return result

    structure = jax.eval_shape(evaluation_at, x_aval, p_aval)
    if structure.objective.shape != ():
        raise ValueError(
            f"The objective must be a scalar, not shape {structure.objective.shape}."
        )
    constraints = tree_layout(structure.constraints)
    count = constraints.size

    def evaluate(x: Any, p: Any) -> Any:
        result = evaluation_at(x, p)
        objective = sign * jnp.asarray(result.objective).astype(jnp.float64)
        parts = [
            jnp.ravel(jnp.asarray(array).astype(jnp.float64))
            for array in jax.tree.leaves(result.constraints)
        ]
        return jnp.concatenate([objective[None], *parts])

    def lagrangian(x: Any, p: Any, factor: Any, multipliers: Any) -> Any:
        values = evaluate(x, p)
        return factor * values[0] + jnp.dot(multipliers[:count], values[1:])

    return FlatFunctions(evaluate, constraints, lagrangian)


def sample_arguments(
    variables: TreeLayout, params: ParamsLayout, multiplier_size: int
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
    variables: TreeLayout,
    params: ParamsLayout,
    multiplier_size: int,
    options: CompileOptions,
) -> DerivativeFunctions:
    """Returns the derivatives of a problem with their patterns."""
    asdex = load_asdex(options)
    samples = sample_arguments(variables, params, multiplier_size)
    n = variables.size
    m = functions.constraint_size

    given = (
        None
        if problem.jacobian_pattern is None or m == 0
        else as_entries(problem.jacobian_pattern, (m, n))
    )
    first_order = first_order_derivative(
        functions.evaluate, given, samples[:2], (m, n), asdex
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
    return DerivativeFunctions(first_order, hessian)


def first_order_derivative(
    evaluate: FlatFunction,
    given: Entries | None,
    samples: tuple[Any, ...],
    shape: tuple[int, int],
    asdex: Any | None,
) -> Derivative:
    """Returns the objective, gradient, constraints and Jacobian of a stacked evaluation.

    `evaluate` returns the objective and then the `m` constraint values, so its Jacobian has
    `1 + m` rows and the first one is the gradient. One transformation of the whole function
    gives the values and the derivatives, so a computation that the objective and the
    constraints share is differentiated once, and the values come out of the same pass. The
    rows and columns of the result are those of the Jacobian of the constraints.
    """
    m, n = shape
    if asdex is None:
        return dense_first_order(evaluate, given, shape)
    if not hasattr(asdex, "value_and_jacobian_from_coloring"):
        return separate_first_order(evaluate, given, samples, shape, asdex)

    # The objective row of the stacked Jacobian is dense, and only the constraints need
    # their sparsity detected. Detection would fail for an objective that asdex cannot
    # trace, and the gradient does not need it.
    if given is None and m > 0:

        def constraints(x: Any, p: Any) -> Any:
            return evaluate(x, p)[1:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", asdex.DenseColoringWarning)
            detected = colored_pattern(
                asdex, "jacobian", constraints, None, samples, (m, n), "the Jacobian"
            )
        given = entries_of(detected.sparsity, (m, n))
    constraint_rows = np.zeros(0, np.intc) if given is None else given.rows
    constraint_columns = np.zeros(0, np.intc) if given is None else given.columns
    stacked = Entries(
        np.concatenate([np.zeros(n, np.intc), constraint_rows + 1]).astype(np.intc),
        np.concatenate([np.arange(n, dtype=np.intc), constraint_columns]).astype(
            np.intc
        ),
        (m + 1, n),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        coloring = colored_pattern(
            asdex, "jacobian", evaluate, stacked, samples, (m + 1, n), "the Jacobian"
        )
    evaluate_with_value = asdex.value_and_jacobian_from_coloring(evaluate, coloring)
    entries = entries_of(coloring.sparsity, (m + 1, n))
    objective_positions = np.flatnonzero(entries.rows == 0)
    constraint_positions = np.flatnonzero(entries.rows > 0)
    objective_columns = jnp.asarray(entries.columns[objective_positions])
    rows = (entries.rows[constraint_positions] - 1).astype(np.intc)
    columns = entries.columns[constraint_positions]

    def sparse(x: Any, p: Any) -> Any:
        values, jacobian = evaluate_with_value(x, p)
        data = jacobian.data
        gradient = jnp.zeros(n).at[objective_columns].set(data[objective_positions])
        return values[0], gradient, values[1:], data[constraint_positions]

    return Derivative(sparse, rows, columns)


def dense_first_order(
    evaluate: FlatFunction, given: Entries | None, shape: tuple[int, int]
) -> Derivative:
    """Returns the first-order function from JAX's dense Jacobian of the stacked evaluation."""
    m, n = shape
    entries = given if given is not None else dense_entries(shape)
    rows, columns = jnp.asarray(entries.rows), jnp.asarray(entries.columns)
    differentiate = jax.jacfwd if n <= m + 1 else jax.jacrev

    def with_values(x: Any, p: Any) -> Any:
        values = evaluate(x, p)
        return values, values

    def dense(x: Any, p: Any) -> Any:
        jacobian, values = differentiate(with_values, has_aux=True)(x, p)
        return values[0], jacobian[0], values[1:], jacobian[1:][rows, columns]

    return Derivative(dense, entries.rows, entries.columns)


def separate_first_order(
    evaluate: FlatFunction,
    given: Entries | None,
    samples: tuple[Any, ...],
    shape: tuple[int, int],
    asdex: Any,
) -> Derivative:
    """Returns the first-order function for a version of asdex that cannot return values.

    The gradient and the Jacobian come from two transformations, so a computation that the
    objective and the constraints share runs for each of them.
    """

    def objective(x: Any, p: Any) -> Any:
        return evaluate(x, p)[0]

    def constraints(x: Any, p: Any) -> Any:
        return evaluate(x, p)[1:]

    jacobian = jacobian_derivative(constraints, given, samples, shape, asdex)

    def separate(x: Any, p: Any) -> Any:
        value, gradient = jax.value_and_grad(objective)(x, p)
        return value, gradient, constraints(x, p), jacobian.function(x, p)

    return Derivative(separate, jacobian.rows, jacobian.columns)


def jacobian_derivative(
    function: FlatFunction,
    given: Entries | None,
    samples: tuple[Any, ...],
    shape: tuple[int, int],
    asdex: Any,
) -> Derivative:
    """Returns the Jacobian of a function of `(x, p)` at the entries of its pattern.

    Only the fallback for a version of `asdex` that cannot return values calls this, so
    `asdex` is always there.
    """
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
    variables: TreeLayout, params: ParamsLayout, multiplier_size: int
) -> ProgramArguments:
    """Returns the abstract arguments of the programs."""
    x = jax.ShapeDtypeStruct((variables.size,), jnp.float64)
    p = jax.ShapeDtypeStruct((params.size,), jnp.float64)
    factor = jax.ShapeDtypeStruct((), jnp.float64)
    multipliers = jax.ShapeDtypeStruct((multiplier_size,), jnp.float64)
    return ProgramArguments((x, p), (x, p, factor, multipliers))
