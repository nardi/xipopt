"""Sparsity patterns as sorted lists of entries."""

from typing import Any, NamedTuple

import numpy as np
from jax.experimental import sparse as jax_sparse
from jaxtyping import Int

from xipopt._problem_types import Pattern


class Entries(NamedTuple):
    """The nonzero entries of a matrix in row-major order, without duplicates."""

    rows: Int[np.ndarray, " nnz"]
    columns: Int[np.ndarray, " nnz"]
    shape: tuple[int, int]


def entries_from_coordinates(
    rows: Any, columns: Any, shape: tuple[int, int]
) -> Entries:
    """Returns the sorted entries at the given coordinates, checking them against the shape."""
    row_array = np.asarray(rows).astype(np.int64).ravel()
    column_array = np.asarray(columns).astype(np.int64).ravel()
    if row_array.shape != column_array.shape:
        raise ValueError("The rows and columns of a pattern must have the same length.")
    outside = (
        (row_array < 0)
        | (row_array >= shape[0])
        | (column_array < 0)
        | (column_array >= shape[1])
    )
    if np.any(outside):
        raise ValueError(f"A pattern has entries outside of the shape {shape}.")
    # A key per entry sorts them row-major and removes duplicates in one step.
    keys = np.unique(row_array * shape[1] + column_array)
    return Entries(
        (keys // shape[1]).astype(np.intc), (keys % shape[1]).astype(np.intc), shape
    )


def check_shape(actual: tuple[int, ...], expected: tuple[int, int]) -> None:
    """Raises a `ValueError` unless a pattern has the shape of the matrix it describes."""
    if tuple(actual) != expected:
        raise ValueError(f"A pattern has shape {tuple(actual)}, expected {expected}.")


def as_entries(pattern: Pattern, shape: tuple[int, int]) -> Entries:
    """Returns the entries of a pattern in any of the accepted forms.

    The pattern is checked against the shape of the matrix it describes.
    """
    match pattern:
        case jax_sparse.BCOO():
            check_shape(pattern.shape, shape)
            indices = np.asarray(pattern.indices).reshape(-1, 2)
            return entries_from_coordinates(indices[:, 0], indices[:, 1], shape)
        case (rows, columns) if isinstance(pattern, tuple):
            return entries_from_coordinates(rows, columns, shape)
        case _ if hasattr(pattern, "rows") and hasattr(pattern, "cols"):
            # An `asdex.SparsityPattern`, which is not imported because asdex is optional.
            check_shape(pattern.shape, shape)
            return entries_from_coordinates(pattern.rows, pattern.cols, shape)
        case _:
            matrix = np.asarray(pattern)
            check_shape(matrix.shape, shape)
            rows, columns = np.nonzero(matrix)
            return entries_from_coordinates(rows, columns, shape)


def dense_entries(shape: tuple[int, int]) -> Entries:
    """Returns the entries of a matrix that is nonzero everywhere."""
    rows, columns = np.indices(shape).reshape(2, -1)
    return entries_from_coordinates(rows, columns, shape)


def symmetric_entries(entries: Entries) -> Entries:
    """Returns the entries together with their mirror images."""
    return entries_from_coordinates(
        np.concatenate([entries.rows, entries.columns]),
        np.concatenate([entries.columns, entries.rows]),
        entries.shape,
    )


def lower_triangle(entries: Entries) -> Entries:
    """Returns the entries on and below the diagonal, with those above it mirrored down."""
    return entries_from_coordinates(
        np.maximum(entries.rows, entries.columns),
        np.minimum(entries.rows, entries.columns),
        entries.shape,
    )


def pattern_digest(entries: Entries | None) -> bytes:
    """Returns bytes that identify a pattern, for use in cache keys."""
    if entries is None:
        return b"none"
    return (
        entries.rows.tobytes()
        + b","
        + entries.columns.tobytes()
        + repr(entries.shape).encode()
    )
