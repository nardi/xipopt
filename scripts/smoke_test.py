"""Solves a small problem with an installed xipopt, to check that a wheel works."""

import jax.numpy as jnp
import numpy as np

import xipopt as xip


def main() -> None:
    """Solves problem 71 of Hock and Schittkowski and checks the known optimum."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(4,)),
        evaluate=xip.evaluation_from_functions(
            objective=lambda x, params: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
            constraints=(
                lambda x, params: jnp.prod(x),
                lambda x, params: jnp.sum(x**2),
            ),
        ),
    )
    solution = xip.solve(
        problem,
        np.array([1.0, 5.0, 5.0, 1.0]),
        variable_bounds=xip.Bounds(1.0, 5.0),
        constraint_bounds=xip.Bounds((25.0, 40.0), (np.inf, 40.0)),
    )
    assert solution.status is xip.Status.SOLVE_SUCCEEDED, solution.status
    assert abs(solution.objective - 17.0140173) < 1e-6, solution.objective
    print(
        f"Ipopt {xip.ipopt_version()} solved problem 71 in {solution.iterations} iterations."
    )


if __name__ == "__main__":
    main()
