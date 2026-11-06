"""Descriptions of optimization problems over pytrees of variables."""

import dataclasses
import math
from typing import Any, NamedTuple, Protocol, Self, overload

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, PyTree

from xipopt._types import Sense


class ParametricFunction(Protocol):
    """A JAX function `f(x, params)` of the variables and the parameters of a problem.

    `x` has the structure of `Problem.variables` with an array in place of every `Variable`.
    `params` is the pytree passed to `compile` and `solve`.
    """

    def __call__(self, x: Any, params: Any, /) -> Float[Array, "*shape"]: ...


class Variable(NamedTuple):
    """An array of continuous decision variables.

    A variable has a shape and no bounds. The bounds of a solve go to
    [`solve`][xipopt.solve] as `variable_bounds`.
    """

    shape: tuple[int, ...] = ()


class Bounds(NamedTuple):
    """The bounds `lower <= value <= upper` of the variables or of the constraints.

    `lower` and `upper` are each a number or an array, which broadcasts to every array, or a
    pytree with the structure of the variables or of the constraints, whose leaves broadcast
    to the shape of the array they belong to. Alternatively, `solve` accepts a pytree with
    that structure and a `Bounds` at every leaf, where the `lower` and `upper` of a leaf
    broadcast to the array at its position. Equal bounds make an equality, and `-inf` and
    `inf` mean no bound.
    """

    lower: PyTree[ArrayLike] = -math.inf
    upper: PyTree[ArrayLike] = math.inf

    @classmethod
    def from_array(cls, bounds: PyTree[ArrayLike]) -> Self:
        """Returns the bounds from a pytree of arrays whose last axis has length two.

        `leaf[..., 0]` is the lower and `leaf[..., 1]` the upper bound of the leaf.
        """

        def lower_of(leaf: ArrayLike) -> np.ndarray:
            return np.asarray(leaf)[..., 0]

        def upper_of(leaf: ArrayLike) -> np.ndarray:
            return np.asarray(leaf)[..., 1]

        return cls(jax.tree.map(lower_of, bounds), jax.tree.map(upper_of, bounds))


BoundsTree = Bounds | PyTree[Bounds]
"""Bounds as one `Bounds`, or as a pytree with the structure of the variables or of the
constraints and a `Bounds` at every leaf."""


class Evaluation(NamedTuple):
    """What a problem computes at a point: the objective and the values of all constraints."""

    objective: Float[Array, ""]
    """The scalar objective. A problem without an objective returns `jnp.zeros(())`."""
    constraints: PyTree[Float[Array, "*shape"]] = ()
    """A pytree of arrays, whose structure is the structure of `Solution.constraints` and of
    the constraint bounds. Every entry of every array is one constraint."""


class EvaluationFunction(Protocol):
    """A JAX function `f(x, params)` that returns the objective and the constraints together.

    `x` has the structure of `Problem.variables` with an array in place of every `Variable`.
    `params` is the pytree passed to `compile` and `solve`. Computing both in one function
    lets them share work, such as an inner solve whose result both of them read. xipopt
    evaluates the function once per point and differentiates it once.
    """

    def __call__(self, x: Any, params: Any, /) -> Evaluation: ...


class StatefulEvaluationFunction(Protocol):
    """A JAX function `f(x, params, state)` that also reads and returns a state.

    `state` has the structure of `StatefulProblem.state`. The function returns the
    evaluation and the new state, with the same structure and shapes. The state never enters
    a derivative, so it may only choose where an inner iteration starts and not what it
    converges to.
    """

    def __call__(
        self, x: Any, params: Any, state: Any, /
    ) -> tuple[Evaluation, Any]: ...


Pattern = Any
"""Where a derivative can be nonzero, given in one of four forms.

- A boolean matrix, `Bool[np.ndarray, "rows columns"]`.
- A pair `(rows, columns)` of integer vectors, one entry per nonzero.
- An `asdex.SparsityPattern`, which has to carry its `input_avals` when it is built.
- A `jax.experimental.sparse.BCOO` matrix, whose stored entries are the nonzeros.

The rows of a Jacobian pattern are the entries of the constraints in the order of the leaves
of `Evaluation.constraints`, and the columns are the flattened variables. A Hessian pattern
is square over the flattened variables and may list one triangle or both.
"""


@dataclasses.dataclass(frozen=True, init=False)
class Problem:
    """A nonlinear optimization problem over a pytree of variables.

    The problem is given in one of two forms. A function `evaluate` returns the objective
    and all constraint values together, so that they can share work such as an inner solve.
    Or separate functions `objective` and `constraints` are given, and the problem builds
    `evaluate` from them. The bounds of the variables and the constraints are not part of
    the problem. They go to `solve`, which can solve one problem for several sets of bounds.
    """

    variables: PyTree[Variable]
    """A pytree with a `Variable` at every leaf, which fixes the structure of `x`."""
    evaluate: EvaluationFunction
    """The function that returns the objective and the constraint values."""
    sense: Sense
    jacobian_pattern: Pattern | None
    """Where the Jacobian of the constraints is nonzero, or `None` to detect it."""
    hessian_pattern: Pattern | None
    """Where the Hessian of the Lagrangian is nonzero, or `None` to detect it."""

    @overload
    def __init__(
        self,
        variables: PyTree[Variable],
        evaluate: EvaluationFunction,
        *,
        sense: Sense = Sense.MINIMIZE,
        jacobian_pattern: Pattern | None = None,
        hessian_pattern: Pattern | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        variables: PyTree[Variable],
        *,
        objective: ParametricFunction | None = None,
        constraints: PyTree[ParametricFunction] = (),
        sense: Sense = Sense.MINIMIZE,
        jacobian_pattern: Pattern | None = None,
        hessian_pattern: Pattern | None = None,
    ) -> None: ...

    def __init__(
        self,
        variables: PyTree[Variable],
        evaluate: EvaluationFunction | None = None,
        *,
        objective: ParametricFunction | None = None,
        constraints: PyTree[ParametricFunction] = (),
        sense: Sense = Sense.MINIMIZE,
        jacobian_pattern: Pattern | None = None,
        hessian_pattern: Pattern | None = None,
    ) -> None:
        """Builds a problem from `evaluate`, or from `objective` and `constraints`.

        Giving `evaluate` together with `objective` or `constraints` is a `TypeError`, and
        so is giving none of them. A problem without an `objective` only needs a feasible
        point.
        """
        has_functions = objective is not None or bool(jax.tree.leaves(constraints))
        if evaluate is not None and has_functions:
            raise TypeError(
                "Give either `evaluate`, or `objective` and `constraints`, but not both."
            )
        if evaluate is None:
            if not has_functions:
                raise TypeError(
                    "Give `evaluate`, or at least one of `objective` and `constraints`."
                )
            evaluate = evaluation_from_functions(
                objective=no_objective if objective is None else objective,
                constraints=constraints,
            )
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "evaluate", evaluate)
        object.__setattr__(self, "sense", sense)
        object.__setattr__(self, "jacobian_pattern", jacobian_pattern)
        object.__setattr__(self, "hessian_pattern", hessian_pattern)


class StatefulProblem(NamedTuple):
    """A `Problem` whose evaluation carries a state from one point to the next.

    xipopt keeps the state between evaluations. Every successful evaluation of the objective
    and constraints replaces it by the state that `evaluate` returned. A solve starts from
    `state`, or from the state of the `Solution` that it warm starts from. Results then
    depend on the order of evaluations, so use the state only for the starting guess of an
    inner solve that converges to its tolerance, such as a Newton iteration. A `Problem` has
    no state and evaluates every point the same way.
    """

    variables: PyTree[Variable]
    """A pytree with a `Variable` at every leaf, which fixes the structure of `x`."""
    evaluate: StatefulEvaluationFunction
    """The function that returns the evaluation and the new state."""
    state: PyTree[ArrayLike]
    """The initial state, a nonempty pytree of floating point arrays."""
    sense: Sense = Sense.MINIMIZE
    jacobian_pattern: Pattern | None = None
    """Where the Jacobian of the constraints is nonzero, or `None` to detect it."""
    hessian_pattern: Pattern | None = None
    """Where the Hessian of the Lagrangian is nonzero, or `None` to detect it."""


def evaluation_from_functions(
    *, objective: ParametricFunction, constraints: PyTree[ParametricFunction] = ()
) -> EvaluationFunction:
    """Returns an evaluation function from a separate objective and separate constraints.

    Each function takes `(x, params)`. `constraints` is a pytree of functions, such as a
    tuple or a dict, and the arrays they return have that structure in
    `Evaluation.constraints`. The functions run on the same arguments, so work that they do
    not share is done once for each. A problem that wants to share work writes its own
    `evaluate`. A problem without an objective passes `no_objective`.
    """

    def evaluate(x: Any, params: Any, /) -> Evaluation:
        return Evaluation(
            objective(x, params),
            jax.tree.map(lambda function: function(x, params), constraints),
        )

    return evaluate


def no_objective(x: Any, params: Any, /) -> Float[Array, ""]:
    """Returns the constant zero, as the objective of a problem that only needs a feasible point."""
    return jnp.zeros(())


def is_variable(value: object) -> bool:
    """Returns whether a value is a `Variable`, for use as the `is_leaf` of tree functions."""
    return isinstance(value, Variable)
