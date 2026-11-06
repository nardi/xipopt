"""Bounds of the variables and the constraints as flat vectors for Ipopt."""

from typing import Any

import jax
import numpy as np
from jaxtyping import Float, PyTree

from xipopt._layout import INFINITY, TreeLayout
from xipopt._problem_types import Bounds


def is_bounds(value: Any) -> bool:
    """Returns whether a node of a pytree is a `Bounds`, which is a pytree node itself."""
    return isinstance(value, Bounds)


def transposed_bounds(bounds: Any, layout: TreeLayout) -> Bounds | None:
    """Returns the `Bounds` of pytrees for a pytree of `Bounds`, and `None` for any other value.

    The pytree has the structure of the layout and a `Bounds` at every leaf. The result has
    that structure in its `lower` and in its `upper`, with the numbers or arrays of each
    leaf.
    """
    if is_bounds(bounds):
        return None
    structure = jax.tree.structure(bounds, is_leaf=is_bounds)
    leaves = jax.tree.leaves(bounds, is_leaf=is_bounds)
    if (
        structure != layout.structure
        or not leaves
        or not all(is_bounds(leaf) for leaf in leaves)
    ):
        return None
    return Bounds(
        jax.tree.map(lambda leaf: leaf.lower, bounds, is_leaf=is_bounds),
        jax.tree.map(lambda leaf: leaf.upper, bounds, is_leaf=is_bounds),
    )


def broadcast_bound(
    value: PyTree[Any], layout: TreeLayout, argument: str, what: str
) -> Float[np.ndarray, " n"]:
    """Returns one side of the bounds as a flat vector over the entries of the layout.

    A number or an array broadcasts to every leaf. A pytree needs the structure of the
    layout, and each of its leaves broadcasts to the shape of the leaf at the same path.
    """
    structure = jax.tree.structure(value)
    if structure == layout.structure:
        leaves = jax.tree.leaves(value)
    elif structure.num_nodes == 1:
        leaves = [value] * len(layout.shapes)
    else:
        raise ValueError(
            f"`{argument}` has structure {structure}, but the {what} have "
            f"{layout.structure}."
        )
    parts = []
    for leaf, shape in zip(leaves, layout.shapes, strict=True):
        try:
            parts.append(np.broadcast_to(np.asarray(leaf, np.float64), shape).ravel())
        except ValueError as error:
            raise ValueError(
                f"A bound in `{argument}` does not broadcast to the shape {shape} of an "
                f"array of the {what}."
            ) from error
    if not parts:
        return np.zeros(0)
    return np.clip(np.concatenate(parts), -INFINITY, INFINITY)


def bounds_vectors(
    bounds: Any, layout: TreeLayout, argument: str, what: str
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
    """Returns the lower and upper bounds as flat vectors, unbounded for `None`.

    `bounds` is a `Bounds`, any pair of lower and upper bounds, or a pytree of `Bounds` with
    the structure of the layout. A lower bound above an upper bound is a `ValueError`.
    """
    if bounds is None:
        return np.full(layout.size, -INFINITY), np.full(layout.size, INFINITY)
    transposed = transposed_bounds(bounds, layout)
    if transposed is not None:
        bounds = transposed
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError(
            f"`{argument}` must be a `Bounds`, or a pytree with the structure of the "
            f"{what} and a `Bounds` at every leaf."
        )
    lower, upper = bounds
    lower_vector = broadcast_bound(lower, layout, f"{argument}.lower", what)
    upper_vector = broadcast_bound(upper, layout, f"{argument}.upper", what)
    if np.any(lower_vector > upper_vector):
        raise ValueError(f"`{argument}` has a lower bound above an upper bound.")
    return lower_vector, upper_vector


def start_point(
    lower: Float[np.ndarray, " n"], upper: Float[np.ndarray, " n"]
) -> Float[np.ndarray, " n"]:
    """Returns the point nearest to the origin inside the bounds."""
    return np.clip(np.zeros(lower.shape[0]), lower, upper)
