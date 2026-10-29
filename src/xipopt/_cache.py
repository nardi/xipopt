"""The disk cache for the programs of a problem.

Building a problem has slow steps, detecting sparsity, lowering the programs to StableHLO and
compiling them. The cache of cyxla stores the lowered programs and the executables. This
module adds the sparsity structure that belongs to the programs and the key under which both
are stored.
"""

import importlib.metadata
from pathlib import Path
from typing import Any, NamedTuple

import cyxla
import jax
import numpy as np
from jaxtyping import Int

from xipopt._options import CompileOptions
from xipopt._patterns import Entries, pattern_digest
from xipopt._types import BuildStatistics

STRUCTURE_SUFFIX = ".xipopt.npz"
"""Suffix of the files with the sparsity structure of a problem."""


class Structure(NamedTuple):
    """Where the Jacobian and the Hessian of a problem have entries."""

    jacobian_rows: Int[np.ndarray, " nnz"]
    jacobian_columns: Int[np.ndarray, " nnz"]
    hessian_rows: Int[np.ndarray, " nnz"]
    hessian_columns: Int[np.ndarray, " nnz"]


def key_distributions() -> tuple[str, ...]:
    """Returns the distributions besides JAX and cyxla whose code determines the programs."""
    names = ["xipopt"]
    try:
        importlib.metadata.distribution("asdex")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        names.append("asdex")
    return tuple(names)


def jaxpr_digest(function: Any, arguments: tuple[Any, ...]) -> bytes:
    """Returns bytes that identify a function by its jaxpr and the constants it captures.

    The text of a jaxpr names its constants but not their values, so the values are added.
    """
    closed = jax.make_jaxpr(function)(*arguments)
    parts = [str(closed.jaxpr).encode()]
    parts += [np.asarray(constant).tobytes() for constant in closed.consts]
    return b"\0".join(parts)


def problem_key(
    objective: bytes,
    constraints: bytes,
    options: CompileOptions,
    sense: int,
    jacobian_pattern: Entries | None,
    hessian_pattern: Entries | None,
) -> str:
    """Returns the cache key of the programs of a problem."""
    return cyxla.cache_key(
        objective,
        constraints,
        repr(options),
        repr(sense),
        pattern_digest(jacobian_pattern),
        pattern_digest(hessian_pattern),
        distributions=key_distributions(),
    )


def structure_path(key: str) -> Path:
    """Returns the path of the structure file for a key."""
    return cyxla.cache_directory() / f"{key}{STRUCTURE_SUFFIX}"


def store_structure(key: str, structure: Structure) -> None:
    """Stores the structure of a problem under a key."""
    path = structure_path(key)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **structure._asdict())
    temporary.replace(path)


def load_structure(key: str) -> Structure | None:
    """Returns the structure stored under a key, or `None` if there is none.

    A file that cannot be read is removed.
    """
    path = structure_path(key)
    if not path.exists():
        return None
    try:
        with np.load(path) as archive:
            return Structure(**{name: archive[name] for name in Structure._fields})
    except (OSError, ValueError, KeyError):
        path.unlink(missing_ok=True)
        return None


def combine(
    statistics: cyxla.CompileStatistics,
    detect: float,
    lower: float,
    lowered: int,
    wall: float,
) -> BuildStatistics:
    """Returns the statistics of a build from those of the compiler and the other steps."""
    return BuildStatistics(
        detect_seconds=detect,
        lower_seconds=lower,
        compile_seconds=statistics.compile_seconds,
        load_seconds=statistics.load_seconds,
        lowered=lowered,
        compiled=statistics.compiled,
        loaded=statistics.loaded,
        wall_seconds=wall,
    )
