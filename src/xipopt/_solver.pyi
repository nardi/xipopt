from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
from cyxla import Executable
from jaxtyping import Float, Int

class SolverError(RuntimeError): ...

class IterationState(NamedTuple):
    mode: int
    iteration: int
    objective: float
    primal_infeasibility: float
    dual_infeasibility: float
    barrier_parameter: float
    step_norm: float
    regularization_size: float
    dual_step_size: float
    primal_step_size: float
    line_search_trials: int
    variables: Float[np.ndarray, " n"]

class NativeResult(NamedTuple):
    status: int
    objective: float
    variables: Float[np.ndarray, " n"]
    constraints: Float[np.ndarray, " m"]
    constraint_multipliers: Float[np.ndarray, " m"]
    lower_multipliers: Float[np.ndarray, " n"]
    upper_multipliers: Float[np.ndarray, " n"]
    iterations: int
    calls: tuple[int, ...]
    seconds: tuple[float, ...]

class Multipliers(NamedTuple):
    constraints: Float[np.ndarray, " m"]
    lower: Float[np.ndarray, " n"]
    upper: Float[np.ndarray, " n"]

IpoptValue = str | int | float

def forward_blas() -> int: ...
def set_blas_threads(count: int) -> None: ...
def ipopt_version() -> tuple[int, int, int]: ...

class NativeProblem:
    executables: tuple[Executable | None, ...]
    def __init__(
        self,
        executables: Sequence[Executable | None],
        variable_lower: Float[np.ndarray, " n"],
        variable_upper: Float[np.ndarray, " n"],
        constraint_lower: Float[np.ndarray, " m"],
        constraint_upper: Float[np.ndarray, " m"],
        parameter_size: int,
        jacobian_rows: Int[np.ndarray, " nnz"],
        jacobian_columns: Int[np.ndarray, " nnz"],
        hessian_rows: Int[np.ndarray, " nnz"],
        hessian_columns: Int[np.ndarray, " nnz"],
        zero_copy: bool = True,
    ) -> None: ...
    def solve(
        self,
        start: Float[np.ndarray, " n"],
        parameters: Float[np.ndarray, " p"],
        settings: Sequence[tuple[str, IpoptValue]] = (),
        warm_start: Multipliers | None = None,
        observer: Callable[[IterationState], bool] | None = None,
    ) -> NativeResult: ...
