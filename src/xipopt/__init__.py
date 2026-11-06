"""Ipopt for JAX functions, called through the PJRT C API."""

# This import comes first because it prepares the search path for the libraries that the
# extension module loads.
from xipopt import _environment  # noqa: F401
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
    Bounds,
    Evaluation,
    EvaluationFunction,
    ParametricFunction,
    Pattern,
    Problem,
    Variable,
    evaluation_from_functions,
    no_objective,
)
from xipopt._programs import SparsityDetectionError
from xipopt._serialize import dumps, load, loads, save
from xipopt._solve import Solution, solve
from xipopt._solver import forward_blas, ipopt_version
from xipopt._types import (
    BuildStatistics,
    CallbackStatistics,
    IterationInfo,
    Program,
    ProgramStatistics,
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
    "Bounds",
    "Derivatives",
    "Evaluation",
    "EvaluationFunction",
    "Hessian",
    "IpoptSetting",
    "IterationInfo",
    "LinearSolver",
    "ParametricFunction",
    "Pattern",
    "Problem",
    "Program",
    "ProgramStatistics",
    "Sense",
    "Solution",
    "SolveOptions",
    "SolveStatistics",
    "SparsityDetectionError",
    "Status",
    "Variable",
    "compile",
    "dumps",
    "evaluation_from_functions",
    "load",
    "loads",
    "no_objective",
    "save",
    "ipopt_version",
    "solve",
]
