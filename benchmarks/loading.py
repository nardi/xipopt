"""Measures the time to load a saved problem on one thread and on several.

The problem has an inner Newton solve that is unrolled, like the problems of the report this
benchmark follows, so its programs are large and deserializing an executable takes a
noticeable time. The script prints the size of the file and the median load time for each
number of threads.

Usage: python benchmarks/loading.py [newton iterations] [repetitions]
"""

import statistics
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import xipopt as xip

jax.config.update("jax_enable_x64", True)

DIMENSION = 12
"""The size of the linear systems of the inner solve."""


def newton_problem(iterations: int) -> xip.Problem:
    """Returns a problem with three variables and one constraint on an unrolled Newton solve."""
    matrix = 3.0 * np.eye(DIMENSION) + 0.1 * np.ones((DIMENSION, DIMENSION))
    shape = np.linspace(0.0, 1.0, DIMENSION)

    def state(x: Any) -> Any:
        right = x[0] + x[1] * shape + x[2] * shape**2
        y = jnp.zeros(DIMENSION)
        for _ in range(iterations):
            residual = matrix @ y + jnp.sin(y) - right
            y = y - jnp.linalg.solve(matrix + jnp.diag(jnp.cos(y)), residual)
        return y

    def evaluate(x: Any, params: Any) -> Any:
        return jnp.sum(state(x) ** 2) + jnp.sum(x**2)

    def constraint(x: Any, params: Any) -> Any:
        return state(x)[:1] + state(x)[-1:]

    return xip.Problem(
        variables=xip.Variable(shape=(3,)),
        objective=evaluate,
        constraints=(xip.Constraint(constraint, upper=5.0),),
    )


def main() -> None:
    """Saves the problem with and without source locations and times the loading of each."""
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    repetitions = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    problem = newton_problem(iterations)
    for debug_info in (True, False):
        options = xip.CompileOptions(
            disk_cache=False,
            derivatives=xip.Derivatives.DENSE,
            hessian=xip.Hessian.LIMITED_MEMORY,
            debug_info=debug_info,
        )
        start = time.perf_counter()
        compiled = xip.compile(problem, options=options)
        compile_seconds = time.perf_counter() - start
        data = xip.dumps(compiled)
        count = sum(
            executable is not None for executable in compiled.native.executables
        )
        print(
            f"debug_info={debug_info}: compile {compile_seconds:.1f} s, "
            f"file {len(data) / 1e6:.2f} MB, {count} programs"
        )
        for workers in (1, None):
            times = []
            for _ in range(repetitions):
                start = time.perf_counter()
                xip.loads(data, problem.variables, warm_up=False, workers=workers)
                times.append(time.perf_counter() - start)
            label = "default" if workers is None else workers
            print(f"  workers {label}: {statistics.median(times) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
