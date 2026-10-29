# Basic usage

A [`Problem`][xipopt.Problem] has variables, an objective and constraints. Every
function takes the variables `x` and the parameters `params`, in this order, and returns
an array.

## Variables

[`Variable`][xipopt.Variable] describes an array of continuous variables with bounds.
Its shape is the broadcast of `shape` and the shapes of the two bounds, and a missing
bound is infinite. The `variables` of a problem are a pytree with a `Variable` at every
leaf, and `x` has the same structure with an array in place of each leaf.

<!-- name: test_docs_basic -->
```python
import jax.numpy as jnp
import numpy as np

import xipopt as xip

variables = {
    "position": xip.Variable(lower=-5.0, upper=5.0, shape=(2,)),
    "scale": xip.Variable(lower=0.1),
}
```

## Objective and constraints

The objective returns a scalar. Ipopt minimizes it, and `sense=xip.Sense.MAXIMIZE`
maximizes it instead. A problem without an objective asks for a feasible point.

A [`Constraint`][xipopt.Constraint] bounds the output of a function from both sides,
`lower <= function(x, params) <= upper`. The bounds broadcast to the shape of the
output, and every entry of the output is one constraint. Equal bounds make an equality,
and a missing bound is infinite.

<!-- name: test_docs_basic -->
```python
def objective(x, params):
    return jnp.sum((x["position"] - 1.0) ** 2) / x["scale"]


problem = xip.Problem(
    variables=variables,
    objective=objective,
    constraints=(
        xip.Constraint(lambda x, params: jnp.sum(x["position"]), upper=1.0),
        xip.Constraint(
            lambda x, params: x["position"] * x["scale"],
            lower=np.array([-1.0, -2.0]),
            upper=2.0,
        ),
    ),
)
```

## Solving

[`solve`][xipopt.solve] takes the problem and a starting point in the structure of the
variables. Without a starting point it starts at the point nearest to the origin inside
the bounds. Ipopt does not need a feasible start, but it converges faster from a good
one, and a nonconvex problem can have several local solutions.

<!-- name: test_docs_basic -->
```python
start = {"position": np.zeros(2), "scale": np.array(1.0)}
solution = xip.solve(problem, start)
print(solution.status.name, solution.objective)
print(solution.x)
```

<!-- name: test_docs_basic -->
```python
assert solution.status.succeeded
assert solution.x["position"].shape == (2,)
assert solution.constraints[0] <= 1.0 + 1e-6
assert solution.constraints[1].shape == (2,)
```

The [`Solution`][xipopt.Solution] holds the point `x` in the structure of the variables,
the objective in the sense of the problem, the value of each constraint in its own
shape, and the multipliers of the constraints and the bounds. The `status` is Ipopt's
return status. [`Status.succeeded`][xipopt.Status] is true when Ipopt found a point that
meets its tolerances, or the looser acceptable ones. A failure such as
`MAXIMUM_ITERATIONS_EXCEEDED` still returns the last point, so check the status before
trusting `x`.

Ipopt reports the multipliers of the problem it solves, which minimizes the objective. A
maximization therefore has the multipliers of its negated objective.

## Compiling

`solve` accepts a `Problem` and compiles it on the way. Compiling traces the functions,
differentiates them, finds the sparsity of the derivatives and compiles five programs.
Do it once with [`compile`][xipopt.compile] when a problem is solved more than once:

<!-- name: test_docs_basic -->
```python
compiled = xip.compile(problem)
first = xip.solve(compiled, start)
second = xip.solve(compiled, {"position": np.ones(2), "scale": np.array(2.0)})
```

<!-- name: test_docs_basic -->
```python
assert first.status.succeeded and second.status.succeeded
print(compiled.statistics)
```

The [`BuildStatistics`][xipopt.BuildStatistics] of a compiled problem say how long each
step took. The programs run in 64-bit floats whatever the setting of `jax.enable_x64`
is, and xipopt switches that setting on only while it traces and lowers the functions.

## Errors

A function that returns the wrong shape, a constraint whose bounds do not broadcast to
its output, or a Jacobian without entries raises a `ValueError` when the problem
compiles. Ipopt rejecting an option raises a `ValueError` when the solve starts. If a
program fails while Ipopt runs, Ipopt shortens its step, and `solve` raises a
`RuntimeError` with the message of the plugin only when the solve ends without success.
