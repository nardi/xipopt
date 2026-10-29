"""Solves a small problem with an installed xipopt, to check that a wheel works."""

import jax.numpy as jnp
import numpy as np

import xipopt as xip


def main() -> None:
    """Solves problem 71 of Hock and Schittkowski and checks the known optimum."""
    problem = xip.Problem(
        variables=xip.Variable(lower=1.0, upper=5.0, shape=(4,)),
        objective=lambda x, params: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        constraints=(
            xip.Constraint(lambda x, params: jnp.prod(x), lower=25.0),
            xip.Constraint(lambda x, params: jnp.sum(x**2), lower=40.0, upper=40.0),
        ),
    )
    solution = xip.solve(problem, np.array([1.0, 5.0, 5.0, 1.0]))
    assert solution.status is xip.Status.SOLVE_SUCCEEDED, solution.status
    assert abs(solution.objective - 17.0140173) < 1e-6, solution.objective
    print(
        f"Ipopt {xip.ipopt_version()} solved problem 71 in {solution.iterations} iterations."
    )


if __name__ == "__main__":
    main()
