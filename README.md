# xipopt

xipopt solves nonlinear optimization problems written as JAX functions with
[Ipopt](https://coin-or.github.io/Ipopt/).

A problem is a pytree of variables, an objective and constraints, all plain JAX
functions. xipopt compiles the function values and their derivatives into programs for
the [PJRT C API](https://openxla.org/xla/pjrt/pjrt_integration) through
[cyxla](https://nardi.github.io/cyxla/). Ipopt's callbacks execute those programs
directly from Cython, so a solve never enters the Python interpreter. The API follows
[JAX-CIP](https://nardi.github.io/jax-cip/).

## Installation

```bash
pip install xipopt
```

The wheels contain Ipopt with the MUMPS and SPRAL linear solvers, so nothing else needs
installing. They exist for Linux, macOS on Apple silicon and Windows, on Python 3.11 and
later.

## Example

This is problem 71 of Hock and Schittkowski, which the Ipopt documentation uses as its
example.

<!-- name: test_readme -->
```python
import jax.numpy as jnp

import xipopt as xip


def objective(x, params):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


problem = xip.Problem(
    variables=xip.Variable(lower=1.0, upper=5.0, shape=(4,)),
    objective=objective,
    constraints=(
        xip.Constraint(lambda x, params: jnp.prod(x), lower=25.0),
        xip.Constraint(lambda x, params: jnp.sum(x**2), lower=40.0, upper=40.0),
    ),
)
solution = xip.solve(problem, jnp.array([1.0, 5.0, 5.0, 1.0]))
print(solution.status, solution.objective, solution.x)
```

<!-- name: test_readme -->
```python
assert solution.status is xip.Status.SOLVE_SUCCEEDED
assert abs(solution.objective - 17.0140173) < 1e-6
```

## Documentation

The [documentation](https://nardi.github.io/xipopt/) has a user guide, an explanation of
the design and the API reference.

## License

[Hippocratic License 3.0](LICENSE.md). The wheels bundle libraries under their own
licenses, which [licenses](licenses) lists.
