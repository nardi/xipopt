"""Descriptions of optimization problems over pytrees of variables."""

import math
from typing import Any, NamedTuple, Protocol

from jaxtyping import Array, ArrayLike, Float, PyTree

from xipopt._types import Sense


class ParametricFunction(Protocol):
    """A JAX function `f(x, params)` of the variables and the parameters of a problem.

    `x` has the structure of `Problem.variables` with an array in place of every `Variable`.
    `params` is the pytree passed to `compile` and `solve`.
    """

    def __call__(self, x: Any, params: Any, /) -> Float[Array, "*shape"]: ...


class Variable(NamedTuple):
    """An array of continuous decision variables with bounds.

    Its shape is the broadcast of `shape` and the shapes of the bounds.
    """

    lower: ArrayLike = -math.inf
    upper: ArrayLike = math.inf
    shape: tuple[int, ...] = ()


class Constraint(NamedTuple):
    """The constraint `lower ≤ function(x, params) ≤ upper`, one row per output entry.

    `lower` and `upper` broadcast to the shape of the output. Equal bounds make an
    equality constraint.
    """

    function: ParametricFunction
    lower: ArrayLike = -math.inf
    upper: ArrayLike = math.inf


Pattern = Any
"""Where a derivative can be nonzero, given in one of four forms.

- A boolean matrix, `Bool[np.ndarray, "rows columns"]`.
- A pair `(rows, columns)` of integer vectors, one entry per nonzero.
- An `asdex.SparsityPattern`, which has to carry its `input_avals` when it is built.
- A `jax.experimental.sparse.BCOO` matrix, whose stored entries are the nonzeros.

The rows of a Jacobian pattern are the entries of the constraints in the order of
`Problem.constraints`, and the columns are the flattened variables. A Hessian pattern is
square over the flattened variables and may list one triangle or both.
"""


class Problem(NamedTuple):
    """A nonlinear optimization problem over a pytree of variables.

    Without an objective the problem asks for a feasible point.
    """

    variables: PyTree[Variable]
    """A pytree with a `Variable` at every leaf, which fixes the structure of `x`."""
    objective: ParametricFunction | None = None
    """A function with a scalar output."""
    sense: Sense = Sense.MINIMIZE
    constraints: tuple[Constraint, ...] = ()
    jacobian_pattern: Pattern | None = None
    """Where the Jacobian of the constraints is nonzero, or `None` to detect it."""
    hessian_pattern: Pattern | None = None
    """Where the Hessian of the Lagrangian is nonzero, or `None` to detect it."""


def is_variable(value: object) -> bool:
    """Returns whether a value is a `Variable`, for use as the `is_leaf` of tree functions."""
    return isinstance(value, Variable)
