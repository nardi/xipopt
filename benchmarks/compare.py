"""Compares xipopt with cyipopt on the same Ipopt, linear solver and derivative patterns.

cyipopt has to be built against the Ipopt of the wheel, and the xipopt import comes first
so that both use one copy of Ipopt and of its BLAS. The script jits the same flat functions
for cyipopt and gives it the same sparsity structure, and it runs the first-order function
once for each new point as xipopt does, so both sides make the same iterations and differ in
how a callback is evaluated.

Usage: python benchmarks/compare.py [repetitions]
"""

import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import xipopt as xip
from xipopt import _bounds, _layout, _programs

import cyipopt  # isort: skip

jax.config.update("jax_enable_x64", True)

TOLERANCE = 1e-8
MAX_ITERATIONS = 500


class BenchmarkProblem(NamedTuple):
    """A problem with its starting point and the bounds that both solvers use."""

    name: str
    problem: xip.Problem
    start: np.ndarray
    variable_bounds: xip.Bounds | None
    """The bounds of the variables, or `None` for unbounded variables."""
    constraint_bounds: xip.Bounds | None
    """The bounds of the constraints, or `None` for a problem without constraints."""


def hs071() -> BenchmarkProblem:
    """Returns problem 71 of Hock and Schittkowski."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(4,)),
        evaluate=xip.evaluation_from_functions(
            objective=lambda x, p: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
            constraints=(
                lambda x, p: jnp.prod(x),
                lambda x, p: jnp.sum(x**2),
            ),
        ),
    )
    return BenchmarkProblem(
        "HS071",
        problem,
        np.array([1.0, 5.0, 5.0, 1.0]),
        xip.Bounds(1.0, 5.0),
        xip.Bounds((25.0, 40.0), (np.inf, 40.0)),
    )


def chain(size: int) -> BenchmarkProblem:
    """Returns a chained Rosenbrock function with trigonometric equality constraints."""

    def objective(x, p):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    def equations(x, p):
        a, b, c = x[:-2], x[1:-1], x[2:]
        return (
            3.0 * b**3
            + 2.0 * c
            - 5.0
            + jnp.sin(b - c) * jnp.sin(b + c)
            + 4.0 * b
            - a * jnp.exp(a - b)
            - 3.0
        )

    problem = xip.Problem(
        variables=xip.Variable(shape=(size,)),
        evaluate=xip.evaluation_from_functions(
            objective=objective, constraints=equations
        ),
    )
    return BenchmarkProblem(
        f"Chain, n={size}",
        problem,
        np.full(size, 2.0),
        None,
        xip.Bounds(0.0, 0.0),
    )


def bratu(size: int) -> BenchmarkProblem:
    """Returns a control problem whose state solves the Bratu equation."""
    spacing = 1.0 / (size + 1)
    grid = np.linspace(spacing, 1.0 - spacing, size)
    target = 0.5 * grid * (1.0 - grid)

    def objective(x, p):
        return spacing * (jnp.sum((x["u"] - target) ** 2) + 1e-2 * jnp.sum(x["c"] ** 2))

    def equation(x, p):
        padded = jnp.concatenate([jnp.zeros(1), x["u"], jnp.zeros(1)])
        second = (2.0 * x["u"] - padded[:-2] - padded[2:]) / spacing**2
        return second - jnp.exp(x["u"]) - x["c"]

    problem = xip.Problem(
        variables={"u": xip.Variable(shape=(size,)), "c": xip.Variable(shape=(size,))},
        evaluate=xip.evaluation_from_functions(
            objective=objective, constraints=equation
        ),
    )
    return BenchmarkProblem(
        f"Bratu control, n={2 * size}",
        problem,
        np.zeros(2 * size),
        None,
        xip.Bounds(0.0, 0.0),
    )


def median_seconds(run: Callable[[], Any], repetitions: int) -> float:
    """Returns the median wall time of a function after one warm-up call."""
    run()
    times = []
    for _ in range(repetitions):
        begin = time.perf_counter()
        run()
        times.append(time.perf_counter() - begin)
    return statistics.median(times)


class JaxProblem:
    """The callbacks of cyipopt for jitted versions of the flat functions of xipopt."""

    def __init__(self, benchmark: BenchmarkProblem) -> None:
        with jax.enable_x64(True):
            variables = _layout.variable_layout(benchmark.problem.variables)
            layout = _layout.params_layout(None)
            functions = _programs.flat_functions(benchmark.problem, variables, layout)
            count = functions.constraint_size
            derivatives = _programs.build_derivatives(
                benchmark.problem,
                functions,
                variables,
                max(count, 1),
                xip.CompileOptions(),
            )
        self.params = np.zeros(layout.size)
        self.count = count
        # Flat bound vectors built from the same bounds that `xip.solve` receives.
        self.variable_lower, self.variable_upper = _bounds.bounds_vectors(
            benchmark.variable_bounds, variables, "variable_bounds", "variables"
        )
        self.constraint_lower, self.constraint_upper = _bounds.bounds_vectors(
            benchmark.constraint_bounds,
            functions.constraints,
            "constraint_bounds",
            "constraints",
        )
        hessian = derivatives.hessian
        assert hessian is not None
        self.jacobian = derivatives.first_order
        self.hessian = hessian
        self.first_order_function = jax.jit(derivatives.first_order.function)
        self.hessian_function = jax.jit(hessian.function)
        self.variables = variables
        self.point = None
        self.values = None

    def first_order(self, x):
        """Returns the first-order outputs at `x`, run once for each new point like xipopt."""
        if self.point is None or not np.array_equal(self.point, x):
            self.values = tuple(
                np.asarray(value) for value in self.first_order_function(x, self.params)
            )
            self.point = np.array(x)
        return self.values

    def objective(self, x):
        return float(self.first_order(x)[0])

    def gradient(self, x):
        return self.first_order(x)[1]

    def constraints(self, x):
        return self.first_order(x)[2]

    def jacobianstructure(self):
        return self.jacobian.rows, self.jacobian.columns

    def hessianstructure(self):
        return self.hessian.rows, self.hessian.columns

    def jacobian_values(self, x):
        return self.first_order(x)[3]

    def hessian_values(self, x, multipliers, factor):
        return np.asarray(self.hessian_function(x, self.params, factor, multipliers))


def cyipopt_solver(benchmark: BenchmarkProblem) -> Callable[[np.ndarray], Any]:
    """Returns a function that solves the problem with cyipopt from a flat starting point."""
    wrapper = JaxProblem(benchmark)

    class Callbacks:
        def objective(self, x):
            return wrapper.objective(x)

        def gradient(self, x):
            return wrapper.gradient(x)

        def constraints(self, x):
            return wrapper.constraints(x)

        def jacobian(self, x):
            return wrapper.jacobian_values(x)

        def hessian(self, x, multipliers, factor):
            return wrapper.hessian_values(x, multipliers, factor)

        jacobianstructure = staticmethod(wrapper.jacobianstructure)
        hessianstructure = staticmethod(wrapper.hessianstructure)

    nlp = cyipopt.Problem(
        n=wrapper.variables.size,
        m=wrapper.count,
        problem_obj=Callbacks(),
        lb=wrapper.variable_lower,
        ub=wrapper.variable_upper,
        cl=wrapper.constraint_lower,
        cu=wrapper.constraint_upper,
    )
    nlp.add_option("tol", TOLERANCE)
    nlp.add_option("max_iter", MAX_ITERATIONS)
    nlp.add_option("print_level", 0)
    nlp.add_option("sb", "yes")
    nlp.add_option("linear_solver", "mumps")
    return nlp.solve


def solve_with_xipopt(
    compiled: xip.CompiledProblem,
    tree: Any,
    benchmark: BenchmarkProblem,
    options: xip.SolveOptions,
) -> xip.Solution:
    """Solves a compiled problem with the bounds of a benchmark problem."""
    return xip.solve(
        compiled,
        tree,
        options=options,
        variable_bounds=benchmark.variable_bounds,
        constraint_bounds=benchmark.constraint_bounds,
    )


def main() -> None:
    """Runs every problem on both sides and prints a table."""
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    options = xip.SolveOptions(tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)
    rows = []
    for benchmark in [
        hs071(),
        chain(100),
        chain(1000),
        bratu(100),
        bratu(1000),
    ]:
        compiled = xip.compile(benchmark.problem)
        tree = compiled.variables.unflatten(benchmark.start)
        reference = solve_with_xipopt(compiled, tree, benchmark, options)
        xipopt_time = median_seconds(
            lambda: solve_with_xipopt(compiled, tree, benchmark, options), repetitions
        )
        solve = cyipopt_solver(benchmark)
        cyipopt_time = median_seconds(lambda: solve(benchmark.start), repetitions)
        rows.append((benchmark.name, reference.iterations, cyipopt_time, xipopt_time))
    print("| Problem | Iterations | cyipopt | xipopt | Ratio |")
    print("| --- | --- | --- | --- | --- |")
    for name, iterations, slow, fast in rows:
        print(
            f"| {name} | {iterations} | {slow * 1e3:.1f} ms | {fast * 1e3:.1f} ms "
            f"| {slow / fast:.2f}x |"
        )


if __name__ == "__main__":
    main()
