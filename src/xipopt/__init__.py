"""Ipopt for JAX functions, called through the PJRT C API."""

# This import comes first because it prepares the search path for the libraries that the
# extension module loads.
from xipopt import _libraries  # noqa: F401
from xipopt._compile import CompiledProblem, compile  # noqa: A004
from xipopt._options import (
    CompileOptions,
    Derivatives,
    Hessian,
    IpoptSetting,
    LinearSolver,
    SolveOptions,
)
from xipopt._problem_types import (
    Constraint,
    ParametricFunction,
    Pattern,
    Problem,
    Variable,
)
from xipopt._programs import SparsityDetectionError
from xipopt._solve import Solution, solve
from xipopt._solver import forward_blas, ipopt_version
from xipopt._types import (
    BuildStatistics,
    CallbackStatistics,
    IterationInfo,
    Program,
    Sense,
    SolveStatistics,
    Status,
)

forward_blas()

__all__ = [
    "BuildStatistics",
    "CallbackStatistics",
    "CompileOptions",
    "CompiledProblem",
    "Constraint",
    "Derivatives",
    "Hessian",
    "IpoptSetting",
    "IterationInfo",
    "LinearSolver",
    "ParametricFunction",
    "Pattern",
    "Problem",
    "Program",
    "Sense",
    "Solution",
    "SolveOptions",
    "SolveStatistics",
    "SparsityDetectionError",
    "Status",
    "Variable",
    "compile",
    "ipopt_version",
    "solve",
]
