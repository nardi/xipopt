"""The two functions that Ipopt's callbacks evaluate, as flat JAX functions.

Every function takes the flat vector of variables `x` and the flat vector `p`, which
contains the parameters followed by the state of a `StatefulProblem`, if there is one. The
first-order function returns the objective, its gradient, the constraint values and the
values of the Jacobian in one pass, and for a problem with state also the new state. The
Hessian of the Lagrangian also takes the factor of the objective and the multipliers of the
constraints. Derivatives are evaluated only at the entries of a sparsity pattern, in the
order of the rows and columns that come with them.
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
from xipopt._problem_types import Evaluation, Problem, StatefulProblem
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
    """Returns `(values, new_state)` for `(x, p)`. `values` is the objective, negated for a
    problem that maximizes, followed by all constraint values: one vector of length `1 + m`.
    `new_state` is the flat state after the evaluation, or an empty vector for a problem
    without state. For a stateful problem `p` ends with the flat state that goes in."""
    values: FlatFunction
    """Returns only `values` of `evaluate`, which is all that sparsity detection and the
    Hessian need."""
    constraints: TreeLayout
    """The structure and the shapes of the arrays in `Evaluation.constraints`."""
    lagrangian: FlatFunction
    """Returns `factor * objective + multipliers @ constraints` for `(x, p, factor, multipliers)`."""
    parameter_size: int
    """The length of `p`, which is the parameters followed by the state."""
    state_size: int = 0
    """The length of the flat state, zero for a problem without state."""

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
    """Returns `(objective, gradient, constraints, jacobian_values)` for `(x, p)`, followed by
    the new state for a problem with state. The rows and columns are those of the Jacobian
    entries, with the rows counted over the constraints."""
    hessian: Derivative | None


def asdex_installed() -> bool:
    """Returns whether `asdex` can be imported."""
    return importlib.util.find_spec("asdex") is not None


def flat_functions(
    problem: Problem | StatefulProblem,
    variables: TreeLayout,
    params: ParamsLayout,
    state: ParamsLayout | None = None,
) -> FlatFunctions:
    """Returns the functions of a problem on flat vectors.

    The state of a stateful problem is appended to the flat parameters `p`. Derivatives are
    taken with respect to `x` only, so the state is a constant for them, and the new state
    leaves the derivative as an auxiliary value.
    """
    sign = 1.0 if problem.sense is Sense.MINIMIZE else -1.0
    state_size = 0 if state is None else state.size
    parameter_size = params.size + state_size
    x_aval = jax.ShapeDtypeStruct((variables.size,), jnp.float64)
    p_aval = jax.ShapeDtypeStruct((parameter_size,), jnp.float64)

    def evaluation_at(x: Any, p: Any) -> tuple[Evaluation, Any]:
        x_tree = variables.unflatten_traced(x)
        params_tree = params.unflatten_traced(p)
        if state is None:
            assert isinstance(problem, Problem)
            result, new_state = problem.evaluate(x_tree, params_tree), None
        else:
            assert isinstance(problem, StatefulProblem)
            result, new_state = problem.evaluate(
                x_tree, params_tree, state.unflatten_traced(p[params.size :])
            )
        if not isinstance(result, Evaluation):
            raise TypeError(
                f"`evaluate` must return an Evaluation, not {type(result).__name__}."
            )
        return result, new_state

    structure, state_structure = jax.eval_shape(evaluation_at, x_aval, p_aval)
    if structure.objective.shape != ():
        raise ValueError(
            f"The objective must be a scalar, not shape {structure.objective.shape}."
        )
    constraints = tree_layout(structure.constraints)
    if state is not None:
        returned = jax.tree.flatten(state_structure)
        if returned[1] != state.structure or any(
            leaf.shape != aval.shape
            for leaf, aval in zip(returned[0], state.avals, strict=True)
        ):
            raise ValueError(
                "`StatefulProblem.evaluate` must return a state with the structure and "
                "shapes of `StatefulProblem.state`."
            )
    count = constraints.size

    def evaluate(x: Any, p: Any) -> Any:
        result, new_state = evaluation_at(x, p)
        objective = sign * jnp.asarray(result.objective).astype(jnp.float64)
        parts = [
            jnp.ravel(jnp.asarray(array).astype(jnp.float64))
            for array in jax.tree.leaves(result.constraints)
        ]
        flat_state = (
            jnp.zeros(0)
            if new_state is None
            else jnp.concatenate(
                [
                    jnp.ravel(jnp.asarray(leaf)).astype(jnp.float64)
                    for leaf in jax.tree.leaves(new_state)
                ]
            )
        )
        return jnp.concatenate([objective[None], *parts]), flat_state

    def values(x: Any, p: Any) -> Any:
        return evaluate(x, p)[0]

    def lagrangian(x: Any, p: Any, factor: Any, multipliers: Any) -> Any:
        stacked = values(x, p)
        return factor * stacked[0] + jnp.dot(multipliers[:count], stacked[1:])

    return FlatFunctions(
        evaluate, values, constraints, lagrangian, parameter_size, state_size
    )


def sample_arguments(
    variables: TreeLayout, parameter_size: int, multiplier_size: int
) -> tuple[Any, ...]:
    """Returns concrete arguments of the Lagrangian, whose values sparsity detection ignores."""
    return (
        jnp.zeros(variables.size),
        jnp.zeros(parameter_size),
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
    problem: Problem | StatefulProblem,
    functions: FlatFunctions,
    variables: TreeLayout,
    multiplier_size: int,
    options: CompileOptions,
) -> DerivativeFunctions:
    """Returns the derivatives of a problem with their patterns."""
    asdex = load_asdex(options)
    samples = sample_arguments(variables, functions.parameter_size, multiplier_size)
    n = variables.size
    m = functions.constraint_size

    given = (
        None
        if problem.jacobian_pattern is None or m == 0
        else as_entries(problem.jacobian_pattern, (m, n))
    )
    first_order = first_order_derivative(functions, given, samples[:2], (m, n), asdex)

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
    functions: FlatFunctions,
    given: Entries | None,
    samples: tuple[Any, ...],
    shape: tuple[int, int],
    asdex: Any | None,
) -> Derivative:
    """Returns the objective, gradient, constraints and Jacobian of a stacked evaluation.

    `functions.values` returns the objective and then the `m` constraint values, so its
    Jacobian has `1 + m` rows and the first one is the gradient. One transformation of the
    whole function gives the values and the derivatives, so a computation that the objective
    and the constraints share is differentiated once, and the values come out of the same
    pass. The rows and columns of the result are those of the Jacobian of the constraints. A
    problem with state also returns its new state as a fifth output.
    """
    m, n = shape
    evaluate, values, state_size = (
        functions.evaluate,
        functions.values,
        functions.state_size,
    )
    if asdex is None:
        return dense_first_order(evaluate, given, shape, state_size)
    if not hasattr(asdex, "value_and_jacobian_from_coloring"):
        return separate_first_order(functions, given, samples, shape, asdex)

    # The objective row of the stacked Jacobian is dense, and only the constraints need
    # their sparsity detected. Detection would fail for an objective that asdex cannot
    # trace, and the gradient does not need it.
    if given is None and m > 0:

        def constraints(x: Any, p: Any) -> Any:
            return values(x, p)[1:]

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
            asdex, "jacobian", values, stacked, samples, (m + 1, n), "the Jacobian"
        )
    entries = entries_of(coloring.sparsity, (m + 1, n))
    objective_positions = np.flatnonzero(entries.rows == 0)
    constraint_positions = np.flatnonzero(entries.rows > 0)
    objective_columns = jnp.asarray(entries.columns[objective_positions])
    rows = (entries.rows[constraint_positions] - 1).astype(np.intc)
    columns = entries.columns[constraint_positions]

    if state_size > 0:
        # The new state is an auxiliary output of the function, which asdex passes through.
        with_state = asdex.value_and_jacobian_from_coloring(
            evaluate, coloring, has_aux=True
        )

        def sparse_with_state(x: Any, p: Any) -> Any:
            (stacked_values, new_state), jacobian = with_state(x, p)
            data = jacobian.data
            gradient = jnp.zeros(n).at[objective_columns].set(data[objective_positions])
            return (
                stacked_values[0],
                gradient,
                stacked_values[1:],
                data[constraint_positions],
                new_state,
            )

        return Derivative(sparse_with_state, rows, columns)

    evaluate_with_value = asdex.value_and_jacobian_from_coloring(values, coloring)

    def sparse(x: Any, p: Any) -> Any:
        stacked_values, jacobian = evaluate_with_value(x, p)
        data = jacobian.data
        gradient = jnp.zeros(n).at[objective_columns].set(data[objective_positions])
        return (
            stacked_values[0],
            gradient,
            stacked_values[1:],
            data[constraint_positions],
        )

    return Derivative(sparse, rows, columns)


def dense_first_order(
    evaluate: FlatFunction,
    given: Entries | None,
    shape: tuple[int, int],
    state_size: int,
) -> Derivative:
    """Returns the first-order function from JAX's dense Jacobian of the stacked evaluation."""
    m, n = shape
    entries = given if given is not None else dense_entries(shape)
    rows, columns = jnp.asarray(entries.rows), jnp.asarray(entries.columns)
    differentiate = jax.jacfwd if n <= m + 1 else jax.jacrev

    def with_values(x: Any, p: Any) -> Any:
        stacked_values, new_state = evaluate(x, p)
        return stacked_values, (stacked_values, new_state)

    def dense(x: Any, p: Any) -> Any:
        jacobian, (stacked_values, new_state) = differentiate(
            with_values, has_aux=True
        )(x, p)
        result = (
            stacked_values[0],
            jacobian[0],
            stacked_values[1:],
            jacobian[1:][rows, columns],
        )
        return (*result, new_state) if state_size > 0 else result

    return Derivative(dense, entries.rows, entries.columns)


def separate_first_order(
    functions: FlatFunctions,
    given: Entries | None,
    samples: tuple[Any, ...],
    shape: tuple[int, int],
    asdex: Any,
) -> Derivative:
    """Returns the first-order function for a version of asdex that cannot return values.

    The gradient and the Jacobian come from two transformations, so a computation that the
    objective and the constraints share runs for each of them. The new state of a problem
    with state comes out of the objective pass.
    """
    values = functions.values
    evaluate = functions.evaluate
    state_size = functions.state_size

    def objective(x: Any, p: Any) -> Any:
        stacked_values, new_state = evaluate(x, p)
        return stacked_values[0], new_state

    def constraints(x: Any, p: Any) -> Any:
        return values(x, p)[1:]

    jacobian = jacobian_derivative(constraints, given, samples, shape, asdex)

    def separate(x: Any, p: Any) -> Any:
        (value, new_state), gradient = jax.value_and_grad(objective, has_aux=True)(x, p)
        result = (value, gradient, constraints(x, p), jacobian.function(x, p))
        return (*result, new_state) if state_size > 0 else result

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
    # Ipopt uses the lower triangle only, so the program keeps those entries of the symmetric
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

    A given pattern has to include the shapes of the arguments, which `asdex` needs to
    evaluate the compressed derivative.
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
    """The variables and the parameters followed by the state, which the objective, constraints
    and their first derivatives take."""
    second_order: tuple[jax.ShapeDtypeStruct, ...]
    """The arguments of the Hessian of the Lagrangian, which also has the objective factor
    and the multipliers."""


def abstract_arguments(
    variables: TreeLayout, parameter_size: int, multiplier_size: int
) -> ProgramArguments:
    """Returns the abstract arguments of the programs.

    `parameter_size` is the length of the flat parameters followed by the flat state.
    """
    x = jax.ShapeDtypeStruct((variables.size,), jnp.float64)
    p = jax.ShapeDtypeStruct((parameter_size,), jnp.float64)
    factor = jax.ShapeDtypeStruct((), jnp.float64)
    multipliers = jax.ShapeDtypeStruct((multiplier_size,), jnp.float64)
    return ProgramArguments((x, p), (x, p, factor, multipliers))
