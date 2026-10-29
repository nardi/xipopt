"""Options of compiling and solving."""

from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

from xipopt._types import IterationInfo


class Hessian(Enum):
    """How Ipopt gets second derivatives."""

    EXACT = "exact"
    """The Hessian of the Lagrangian is compiled and evaluated at every iteration."""
    LIMITED_MEMORY = "limited-memory"
    """Ipopt approximates the Hessian from gradients, so no Hessian is compiled."""


class Derivatives(Enum):
    """How the Jacobian and the Hessian are computed, and where their patterns come from."""

    AUTOMATIC = "automatic"
    """`asdex` when it is installed, and dense derivatives with a warning when it is not."""
    SPARSE = "sparse"
    """`asdex` detects the patterns, colors them and evaluates the compressed derivatives.

    A pattern that a problem gives replaces the detected one.
    """
    DENSE = "dense"
    """JAX differentiates densely and the programs keep the entries of the pattern.

    A problem without patterns gets all entries, and `asdex` is not needed.
    """


class LinearSolver(Enum):
    """The sparse linear solvers that the Ipopt library of the wheel contains."""

    MUMPS = "mumps"
    SPRAL = "spral"


class CompileOptions(NamedTuple):
    """Options of turning the functions of a problem into programs."""

    hessian: Hessian = Hessian.EXACT
    derivatives: Derivatives = Derivatives.AUTOMATIC
    disk_cache: bool = True
    """Whether lowered programs and executables are stored in the cache directory.

    The directory is `CYXLA_CACHE_DIR` if set, and `~/.cache/cyxla` otherwise.
    """
    zero_copy: bool = True
    """Whether executions read and write host memory directly when the plugin allows it."""


class IpoptSetting(NamedTuple):
    """A value for one Ipopt option."""

    option: str
    """The name of the option, as in Ipopt's documentation."""
    value: str | int | float


class SolveOptions(NamedTuple):
    """Limits and settings of one solve."""

    tolerance: float = 1e-8
    """Ipopt's `tol`, the tolerance on the scaled optimality error."""
    max_iterations: int = 3000
    time_limit: float | None = None
    """Seconds of wall time after which Ipopt stops, or `None` for no limit."""
    linear_solver: LinearSolver = LinearSolver.MUMPS
    print_level: int = 0
    """Ipopt's `print_level` from 0, which prints nothing, to 12."""
    blas_threads: int | None = 1
    """Threads of OpenBLAS, or `None` to leave the setting as it is.

    One thread suits the sparse factorizations of problems that fit in a few million
    entries, since the linear algebra calls are small.
    """
    settings: tuple[IpoptSetting, ...] = ()
    """Further option values, applied after the fields above."""
    callback: Callable[[IterationInfo], bool | None] | None = None
    """A function called at the end of every iteration, or `None`.

    Ipopt stops with `Status.USER_REQUESTED_STOP` when it returns `False`. Returning
    nothing continues. The function runs with the GIL, so it slows a solve down.
    """
