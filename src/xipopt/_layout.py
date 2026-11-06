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


class TreeLayout(NamedTuple):
    """A pytree of arrays, such as the variables or the constraints, as one flat vector."""

    structure: jax.tree_util.PyTreeDef
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]
    """Position of each leaf's first entry in the flat vector."""
    fingerprint: tuple[cyxla.LeafInfo, ...]
    """The path and shape of every leaf, which a tree given to `load` has to match."""

    @property
    def size(self) -> int:
        """Number of scalar entries in all leaves."""
        return sum(math.prod(shape) for shape in self.shapes)

    def flatten(
        self, values: PyTree[Any], name: str = "values"
    ) -> Float[np.ndarray, " n"]:
        """Returns values in the structure of the layout as a flat vector."""
        leaves, structure = jax.tree.flatten(values)
        if structure != self.structure:
            raise ValueError(
                f"The {name} have structure {structure}, but the layout has "
                f"{self.structure}."
            )
        arrays = [np.asarray(leaf, np.float64) for leaf in leaves]
        for array, shape in zip(arrays, self.shapes, strict=True):
            if array.shape != shape:
                raise ValueError(f"A value has shape {array.shape}, expected {shape}.")
        if not arrays:
            return np.zeros(0)
        return np.concatenate([array.ravel() for array in arrays])

    def unflatten(self, values: Float[np.ndarray, " n"]) -> PyTree[np.ndarray]:
        """Returns a flat vector in the structure of the layout."""
        leaves = [
            values[offset : offset + math.prod(shape)].reshape(shape)
            for offset, shape in zip(self.offsets, self.shapes, strict=True)
        ]
        return jax.tree.unflatten(self.structure, leaves)

    def unflatten_traced(self, values: Float[jax.Array, " n"]) -> PyTree[jax.Array]:
        """Returns a traced flat vector in the structure of the layout."""
        leaves = [
            jnp.reshape(values[offset : offset + math.prod(shape)], shape)
            for offset, shape in zip(self.offsets, self.shapes, strict=True)
        ]
        return jax.tree.unflatten(self.structure, leaves)


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

    @property
    def element_count(self) -> int:
        """Number of elements in all leaves, which is zero for an empty tree."""
        return sum(math.prod(aval.shape) for aval in self.avals)

    def unflatten(self, values: Float[np.ndarray, " p"]) -> PyTree[np.ndarray]:
        """Returns a flat vector as arrays in the structure and types of the leaves."""
        leaves = []
        offset = 0
        for aval in self.avals:
            size = math.prod(aval.shape)
            leaf = np.array(values[offset : offset + size]).reshape(aval.shape)
            leaves.append(leaf.astype(aval.dtype))
            offset += size
        return jax.tree.unflatten(self.structure, leaves)

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


def state_layout(state: PyTree[Any]) -> ParamsLayout:
    """Returns the layout of the state of a `StatefulProblem`, which has floating point leaves."""
    layout = params_layout(state)
    if layout.element_count == 0:
        raise ValueError("The state of a StatefulProblem needs at least one element.")
    for aval in layout.avals:
        if not jnp.issubdtype(aval.dtype, jnp.floating):
            raise TypeError(
                f"The leaves of the state must be floating point arrays, not {aval.dtype}."
            )
    return layout


def leaf_shape(leaf: Any) -> tuple[int, ...]:
    """Returns the shape of a leaf, which is a `Variable`, an array or a shape structure."""
    if is_variable(leaf):
        return tuple(leaf.shape)
    if hasattr(leaf, "shape"):
        return tuple(leaf.shape)
    return tuple(np.shape(leaf))


def tree_layout(tree: PyTree[Any], is_leaf: Any = None) -> TreeLayout:
    """Returns the flat layout of a pytree whose leaves have a shape."""
    leaves, structure = jax.tree.flatten(tree, is_leaf=is_leaf)
    shapes = tuple(leaf_shape(leaf) for leaf in leaves)
    sizes = [math.prod(shape) for shape in shapes]
    offsets = [sum(sizes[:index]) for index in range(len(sizes))]
    return TreeLayout(
        structure=structure,
        shapes=shapes,
        offsets=tuple(offsets),
        fingerprint=cyxla.tree_fingerprint(tree, leaf_shape, is_leaf=is_leaf),
    )


def variable_layout(variables: PyTree[Any]) -> TreeLayout:
    """Returns the flat layout of a pytree of `Variable`s."""
    for leaf in jax.tree.leaves(variables, is_leaf=is_variable):
        if not is_variable(leaf):
            raise TypeError(
                f"Every leaf of the variables must be a Variable, not {leaf!r}."
            )
    layout = tree_layout(variables, is_variable)
    if layout.size == 0:
        raise ValueError("A problem needs at least one variable.")
    return layout


def layout_for(
    template: PyTree[Any],
    expected: tuple[cyxla.LeafInfo, ...],
    name: str,
    is_leaf: Any = None,
) -> TreeLayout:
    """Returns the layout of a tree given to `load`, which has to match the saved problem.

    The leaves of the tree are arrays, shape structures or `Variable`s, and only their paths
    and shapes matter. A tree that does not match the fingerprint of the saved problem
    raises a `ValueError` that names the first difference.
    """
    layout = tree_layout(template, is_leaf)
    cyxla.check_fingerprint(expected, layout.fingerprint, name)
    return layout
