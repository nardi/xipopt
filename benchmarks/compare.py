"""Compares xipopt with cyipopt on the same Ipopt, linear solver and derivative patterns.

cyipopt has to be built against the Ipopt of the wheel, and the xipopt import comes first
so that both use one copy of Ipopt and of its BLAS. The script jits the same flat functions
for cyipopt and gives it the same sparsity structure, so both sides make the same
iterations and differ in how a callback is evaluated.

Usage: python benchmarks/compare.py [repetitions]
"""

import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import xipopt as xip
from xipopt import _compile, _layout, _programs

import cyipopt  # isort: skip

jax.config.update("jax_enable_x64", True)

TOLERANCE = 1e-8
MAX_ITERATIONS = 500


def hs071() -> tuple[str, xip.Problem, np.ndarray]:
    """Returns problem 71 of Hock and Schittkowski."""
    problem = xip.Problem(
        variables=xip.Variable(lower=1.0, upper=5.0, shape=(4,)),
        objective=lambda x, p: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        constraints=(
            xip.Constraint(lambda x, p: jnp.prod(x), lower=25.0),
            xip.Constraint(lambda x, p: jnp.sum(x**2), lower=40.0, upper=40.0),
        ),
    )
    return "HS071", problem, np.array([1.0, 5.0, 5.0, 1.0])


def chain(size: int) -> tuple[str, xip.Problem, np.ndarray]:
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
        objective=objective,
        constraints=(xip.Constraint(equations, lower=0.0, upper=0.0),),
    )
    return f"Chain, n={size}", problem, np.full(size, 2.0)


def bratu(size: int) -> tuple[str, xip.Problem, np.ndarray]:
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
        objective=objective,
        constraints=(xip.Constraint(equation, lower=0.0, upper=0.0),),
    )
    return f"Bratu control, n={2 * size}", problem, np.zeros(2 * size)


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

    def __init__(self, problem: xip.Problem) -> None:
        with jax.enable_x64(True):
            variables = _layout.variable_layout(problem.variables)
            layout = _layout.params_layout(None)
            functions = _programs.flat_functions(problem, variables, layout)
            count = functions.constraint_size
            derivatives = _programs.build_derivatives(
                problem,
                functions,
                variables,
                layout,
                max(count, 1),
                xip.CompileOptions(),
            )
        self.params = np.zeros(layout.size)
        self.count = count
        self.constraint_lower, self.constraint_upper = _compile.constraint_bounds(
            problem, functions.constraint_shapes
        )
        self.jacobian = derivatives.jacobian
        self.hessian = derivatives.hessian
        self.objective_function = jax.jit(functions.objective)
        self.gradient_function = jax.jit(derivatives.gradient)
        self.constraint_function = jax.jit(functions.constraints)
        self.jacobian_function = jax.jit(self.jacobian.function)
        self.hessian_function = jax.jit(self.hessian.function)
        self.variables = variables

    def objective(self, x):
        return float(self.objective_function(x, self.params))

    def gradient(self, x):
        return np.asarray(self.gradient_function(x, self.params))

    def constraints(self, x):
        return np.asarray(self.constraint_function(x, self.params))

    def jacobianstructure(self):
        return self.jacobian.rows, self.jacobian.columns

    def hessianstructure(self):
        return self.hessian.rows, self.hessian.columns

    def jacobian_values(self, x):
        return np.asarray(self.jacobian_function(x, self.params))

    def hessian_values(self, x, multipliers, factor):
        return np.asarray(self.hessian_function(x, self.params, factor, multipliers))


def cyipopt_solver(problem: xip.Problem) -> Callable[[np.ndarray], Any]:
    """Returns a function that solves the problem with cyipopt from a flat starting point."""
    wrapper = JaxProblem(problem)

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
        lb=wrapper.variables.lower,
        ub=wrapper.variables.upper,
        cl=wrapper.constraint_lower,
        cu=wrapper.constraint_upper,
    )
    nlp.add_option("tol", TOLERANCE)
    nlp.add_option("max_iter", MAX_ITERATIONS)
    nlp.add_option("print_level", 0)
    nlp.add_option("sb", "yes")
    nlp.add_option("linear_solver", "mumps")
    return nlp.solve


def main() -> None:
    """Runs every problem on both sides and prints a table."""
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    options = xip.SolveOptions(tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)
    rows = []
    for name, problem, start in [
        hs071(),
        chain(100),
        chain(1000),
        bratu(100),
        bratu(1000),
    ]:
        compiled = xip.compile(problem)
        tree = compiled.variables.unflatten(start)
        reference = xip.solve(compiled, tree, options=options)
        xipopt_time = median_seconds(
            lambda: xip.solve(compiled, tree, options=options), repetitions
        )
        solve = cyipopt_solver(problem)
        cyipopt_time = median_seconds(lambda: solve(start), repetitions)
        rows.append((name, reference.iterations, cyipopt_time, xipopt_time))
    print("| Problem | Iterations | cyipopt | xipopt | Ratio |")
    print("| --- | --- | --- | --- | --- |")
    for name, iterations, slow, fast in rows:
        print(
            f"| {name} | {iterations} | {slow * 1e3:.1f} ms | {fast * 1e3:.1f} ms "
            f"| {slow / fast:.2f}x |"
        )


if __name__ == "__main__":
    main()
