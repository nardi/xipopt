# xipopt

xipopt solves nonlinear optimization problems that are written as JAX functions. It
passes them to [Ipopt](https://coin-or.github.io/Ipopt/), an interior point solver for
problems with smooth objectives and constraints.

A problem is a pytree of variables, an objective and constraints. Ipopt wants the
values, gradients, Jacobians and Hessians of those functions many times per solve, and a
solver written in C asks for each of them through a callback. xipopt compiles the
functions and their derivatives with [cyxla](https://nardi.github.io/cyxla/) into
programs for the PJRT C API. Ipopt's callbacks execute those programs directly from
Cython, so a solve does not call back into Python.

The API follows [JAX-CIP](https://nardi.github.io/jax-cip/), which does the same for the
global solver SCIP.

## Installation

```bash
pip install xipopt
```

The [installation page](installation.md) lists the supported platforms and what the
wheels contain.

## A first problem

This is problem 71 of Hock and Schittkowski. The objective is nonlinear, one constraint
is an inequality and the other is an equality.

<!-- name: test_docs_index -->
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
print(solution.status.name, solution.objective)
```

<!-- name: test_docs_index -->
```python
assert solution.status is xip.Status.SOLVE_SUCCEEDED
assert abs(solution.objective - 17.0140173) < 1e-6
```

`solve` compiles the problem first. Compiling takes about a second for a small problem,
and the [disk cache](guide/parameters.md) makes later processes skip most of that.
Compile once with `compile` to solve the same problem again and again.

## What it handles

- Any JAX function that is smooth where Ipopt evaluates it. The functions may use
  control flow such as `lax.scan`, and anything else that cyxla can lower.
- Variables and parameters as pytrees, with bounds on every variable.
- Equality and inequality constraints of any shape, which Ipopt sees as one vector.
- Sparse Jacobians and Hessians, found by [asdex](https://pypi.org/project/asdex/) or
  given as patterns. The [derivatives guide](guide/derivatives.md) explains both.
- Warm starts from the multipliers of an earlier solution.

## What it does not handle

Ipopt finds a local solution of a continuous problem. It has no integer variables, and
it does not prove that its solution is globally optimal.
[JAX-CIP](https://nardi.github.io/jax-cip/) covers mixed-integer problems and global
optimality, at a higher cost per problem.

The programs run on the CPU, because Ipopt's callbacks share memory with the plugin
without copies. There is no GPU path.
