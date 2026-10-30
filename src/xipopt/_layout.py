"""The variables and parameters of a problem as flat vectors."""

import math
from typing import Any, NamedTuple

import cyxla
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, PyTree

from xipopt._problem_types import is_variable

INFINITY = 1e30
"""The value that stands for an infinite bound, as Ipopt treats magnitudes from 1e19 as infinite."""


class VariableLayout(NamedTuple):
    """The variables of a problem as one flat vector."""

    structure: jax.tree_util.PyTreeDef
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]
    """Position of each leaf's first entry in the flat vector."""
    lower: Float[np.ndarray, " n"]
    upper: Float[np.ndarray, " n"]
    fingerprint: tuple[cyxla.LeafInfo, ...]
    """The path and shape of every leaf, which a tree given to `load` has to match."""

    @property
    def size(self) -> int:
        """Number of scalar variables."""
        return self.lower.shape[0]

    def flatten(self, values: PyTree[Any]) -> Float[np.ndarray, " n"]:
        """Returns values in the structure of the variables as a flat vector."""
        leaves, structure = jax.tree.flatten(values)
        if structure != self.structure:
            raise ValueError(
                f"The values have structure {structure}, but the variables have "
                f"{self.structure}."
            )
        arrays = [np.asarray(leaf, np.float64) for leaf in leaves]
        for array, shape in zip(arrays, self.shapes, strict=True):
            if array.shape != shape:
                raise ValueError(f"A value has shape {array.shape}, expected {shape}.")
        return np.concatenate([array.ravel() for array in arrays])

    def unflatten(self, values: Float[np.ndarray, " n"]) -> PyTree[np.ndarray]:
        """Returns a flat vector in the structure of the variables."""
        leaves = [
            values[offset : offset + math.prod(shape)].reshape(shape)
            for offset, shape in zip(self.offsets, self.shapes, strict=True)
        ]
        return jax.tree.unflatten(self.structure, leaves)

    def unflatten_traced(self, values: Float[jax.Array, " n"]) -> PyTree[jax.Array]:
        """Returns a traced flat vector in the structure of the variables."""
        leaves = [
            jnp.reshape(values[offset : offset + math.prod(shape)], shape)
            for offset, shape in zip(self.offsets, self.shapes, strict=True)
        ]
        return jax.tree.unflatten(self.structure, leaves)

    def start(self) -> Float[np.ndarray, " n"]:
        """Returns the point nearest to the origin inside the bounds."""
        return np.clip(np.zeros(self.size), self.lower, self.upper)


class ParamsLayout(NamedTuple):
    """Structure, shapes and types of the parameters a problem was compiled for."""

    structure: jax.tree_util.PyTreeDef
    avals: tuple[jax.ShapeDtypeStruct, ...]
    fingerprint: tuple[cyxla.LeafInfo, ...]
    """The path and shape of every leaf, which a tree given to `load` has to match."""

    @property
    def size(self) -> int:
        """Length of the flat vector, which holds at least one entry so buffers are never empty."""
        return max(sum(math.prod(aval.shape) for aval in self.avals), 1)

    def flatten(self, params: PyTree[Any]) -> Float[np.ndarray, " p"]:
        """Returns the parameters as a flat float64 vector, checking them against the layout."""
        leaves, structure = jax.tree.flatten(params)
        if structure != self.structure:
            raise ValueError(
                f"The parameters have structure {structure}, compiled for {self.structure}."
            )
        arrays = [np.asarray(leaf) for leaf in leaves]
        for array, aval in zip(arrays, self.avals, strict=True):
            if array.shape != aval.shape:
                raise ValueError(
                    f"A parameter has shape {array.shape}, compiled for {aval.shape}."
                )
        flat = np.zeros(self.size)
        if arrays:
            joined = np.concatenate([array.ravel() for array in arrays])
            flat[: joined.shape[0]] = joined
        return flat

    def unflatten_traced(self, values: Float[jax.Array, " p"]) -> PyTree[jax.Array]:
        """Returns a traced flat vector in the structure and types of the parameters."""
        leaves = []
        offset = 0
        for aval in self.avals:
            size = math.prod(aval.shape)
            leaf = jnp.reshape(values[offset : offset + size], aval.shape)
            leaves.append(leaf.astype(aval.dtype))
            offset += size
        return jax.tree.unflatten(self.structure, leaves)


def as_aval(leaf: Any) -> jax.ShapeDtypeStruct:
    """Returns the shape and type of a parameter leaf."""
    if isinstance(leaf, jax.ShapeDtypeStruct):
        return leaf
    array = jnp.asarray(leaf)
    return jax.ShapeDtypeStruct(array.shape, array.dtype)


def params_layout(params: PyTree[Any]) -> ParamsLayout:
    """Returns the layout of parameters, which may be arrays or `jax.ShapeDtypeStruct`s."""
    leaves, structure = jax.tree.flatten(params)
    avals = tuple(as_aval(leaf) for leaf in leaves)
    fingerprint = cyxla.tree_fingerprint(params, lambda leaf: as_aval(leaf).shape)
    return ParamsLayout(structure, avals, fingerprint)


def leaf_shape(leaf: Any) -> tuple[int, ...]:
    """Returns the shape of a leaf of the variables, which is a `Variable` or an array."""
    if is_variable(leaf):
        return np.broadcast_shapes(
            leaf.shape, np.shape(leaf.lower), np.shape(leaf.upper)
        )
    return tuple(np.shape(leaf))


def variable_layout(variables: PyTree[Any]) -> VariableLayout:
    """Returns the flat layout of a pytree of `Variable`s."""
    leaves, structure = jax.tree.flatten(variables, is_leaf=is_variable)
    shapes: list[tuple[int, ...]] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    offsets: list[int] = []
    offset = 0
    for leaf in leaves:
        if not is_variable(leaf):
            raise TypeError(
                f"Every leaf of the variables must be a Variable, not {leaf!r}."
            )
        shape = leaf_shape(leaf)
        lower = np.broadcast_to(np.asarray(leaf.lower, np.float64), shape).ravel()
        upper = np.broadcast_to(np.asarray(leaf.upper, np.float64), shape).ravel()
        if np.any(lower > upper):
            raise ValueError("A variable has a lower bound above its upper bound.")
        shapes.append(shape)
        lowers.append(np.clip(lower, -INFINITY, INFINITY))
        uppers.append(np.clip(upper, -INFINITY, INFINITY))
        offsets.append(offset)
        offset += lower.shape[0]
    if offset == 0:
        raise ValueError("A problem needs at least one variable.")
    return VariableLayout(
        structure=structure,
        shapes=tuple(shapes),
        offsets=tuple(offsets),
        lower=np.concatenate(lowers),
        upper=np.concatenate(uppers),
        fingerprint=cyxla.tree_fingerprint(variables, leaf_shape, is_leaf=is_variable),
    )


def variable_layout_for(
    template: PyTree[Any],
    lower: Float[np.ndarray, " n"],
    upper: Float[np.ndarray, " n"],
    expected: tuple[cyxla.LeafInfo, ...],
) -> VariableLayout:
    """Returns the layout of a tree given to `load`, with the bounds that were saved.

    The leaves of the tree are `Variable`s or arrays, and only their paths and shapes matter.
    A tree that does not match the fingerprint of the saved problem raises a `ValueError`.
    """
    leaves, structure = jax.tree.flatten(template, is_leaf=is_variable)
    fingerprint = cyxla.tree_fingerprint(template, leaf_shape, is_leaf=is_variable)
    cyxla.check_fingerprint(expected, fingerprint, "variables")
    shapes = tuple(leaf_shape(leaf) for leaf in leaves)
    sizes = [math.prod(shape) for shape in shapes]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    return VariableLayout(
        structure=structure,
        shapes=shapes,
        offsets=tuple(int(offset) for offset in offsets),
        lower=lower,
        upper=upper,
        fingerprint=fingerprint,
    )
