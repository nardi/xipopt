# Basic usage

A [`Problem`][xipopt.Problem] has variables and a function that computes the objective
and the constraints. The bounds are not part of the problem. They are passed to
[`solve`][xipopt.solve], as the [bounds section](#bounds) explains. The function takes
the variables `x` and the parameters `params`, in this order.

## Variables

[`Variable`][xipopt.Variable] describes an array of continuous variables by its `shape`.
The `variables` of a problem are a pytree with a `Variable` at every leaf, and `x` has
the same structure with an array in place of each leaf.

<!-- name: test_docs_basic -->
```python
import jax.numpy as jnp
import numpy as np

import xipopt as xip

variables = {
    "position": xip.Variable(shape=(2,)),
    "scale": xip.Variable(),
}
```

## Objective and constraints

`Problem.evaluate` returns an [`Evaluation`][xipopt.Evaluation] with the fields
`objective` and `constraints`. `Evaluation.objective` is the scalar value of the
objective function at `x`. Ipopt minimizes it, and `sense=xip.Sense.MAXIMIZE` maximizes
it instead. A problem that only needs a feasible point returns `jnp.zeros(())` as the
value, or uses [`no_objective`][xipopt.no_objective] as its objective function, and
`solve` then finds a point that satisfies the bounds and the constraints.

`Evaluation.constraints` is a pytree of arrays. Every entry of every array is one
constraint, and the [bounds section](#bounds) explains how to give its bounds.

<!-- name: test_docs_basic -->
```python
def evaluate(x, params):
    objective = jnp.sum((x["position"] - 1.0) ** 2) / x["scale"]
    return xip.Evaluation(
        objective,
        (jnp.sum(x["position"]), x["position"] * x["scale"]),
    )


problem = xip.Problem(variables=variables, evaluate=evaluate)
```

Separate functions for the objective and each constraint also work. Pass them as
`objective` and `constraints` to [`Problem`][xipopt.Problem], which builds `evaluate`
from them with [`evaluation_from_functions`][xipopt.evaluation_from_functions]. Writing
one function lets the objective and the constraints share work, as [the guide on
performance](performance.md) explains.

## Bounds

`solve` takes the bounds of the variables as `variable_bounds` and the bounds of the
constraints as `constraint_bounds`. Each is a [`Bounds`][xipopt.Bounds] with a `lower`
and an `upper`, or a pytree with a `Bounds` at every leaf.

In a `Bounds`, a bound is a number or an array, which broadcasts to every array. A bound
can also be a pytree with the structure of the variables or of `Evaluation.constraints`,
and each leaf of such a pytree broadcasts to the array at the same position. In a pytree
of `Bounds`, the structure is the one of the variables or of `Evaluation.constraints`,
and the `lower` and `upper` of each `Bounds` broadcast to the array at its position.

Equal bounds make an equality, and `-inf` and `inf` mean no bound. A problem with
constraints needs `constraint_bounds`. A problem without constraints must not receive
it. Passing `variable_bounds=None` leaves the variables unbounded.

The bounds are read when a solve starts, so one compiled problem can be solved with many
sets of bounds.

<!-- name: test_docs_basic -->
```python
variable_bounds = xip.Bounds(
    lower={"position": -5.0, "scale": 0.1},
    upper={"position": 5.0, "scale": np.inf},
)
constraint_bounds = xip.Bounds(
    lower=(-np.inf, np.array([-1.0, -2.0])),
    upper=(1.0, 2.0),
)
```

The bounds of the variables can also be written with a `Bounds` for each variable. This
is the same as `variable_bounds` above.

<!-- name: test_docs_basic -->
```python
variable_bounds = {
    "position": xip.Bounds(-5.0, 5.0),
    "scale": xip.Bounds(lower=0.1),
}
```

## Solves

[`solve`][xipopt.solve] takes the problem and a starting point in the structure of the
variables. Without a starting point it starts at the point nearest to the origin inside
the bounds. Ipopt does not need a feasible start, but it converges faster from a good
one, and a nonconvex problem can have several local solutions.

<!-- name: test_docs_basic -->
```python
start = {"position": np.zeros(2), "scale": np.array(1.0)}
solution = xip.solve(
    problem,
    start,
    variable_bounds=variable_bounds,
    constraint_bounds=constraint_bounds,
)
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

The [`Solution`][xipopt.Solution] contains the point `x` in the structure of the
variables, the objective in the sense of the problem, the values of the constraints in
the structure of `Evaluation.constraints`, and the multipliers of the constraints and
the bounds. The `status` is Ipopt's return status. [`Status.succeeded`][xipopt.Status]
is true when Ipopt found a point that meets its tolerances, or the looser acceptable
ones. A failure such as `MAXIMUM_ITERATIONS_EXCEEDED` still returns the last point, so
check the status before trusting `x`.

Ipopt reports the multipliers of the problem it solves, which minimizes the objective. A
maximization therefore has the multipliers of its negated objective.

## Compilation

`solve` accepts a `Problem` and compiles it as part of the call. Compiling traces the
functions, differentiates them and finds the sparsity of the derivatives. It then
compiles two programs. The first-order program returns the objective, its gradient, the
constraints and the Jacobian values in one pass. The Hessian program also takes the
objective factor and the multipliers, which Ipopt supplies only with its Hessian call.
The [design notes](../design.md#programs-and-callbacks) describe how the programs and
the callbacks fit together. Do it once with [`compile`][xipopt.compile] when a problem
is solved more than once:

<!-- name: test_docs_basic -->
```python
compiled = xip.compile(problem)
first = xip.solve(
    compiled,
    start,
    variable_bounds=variable_bounds,
    constraint_bounds=constraint_bounds,
)
second = xip.solve(
    compiled,
    {"position": np.ones(2), "scale": np.array(2.0)},
    variable_bounds=variable_bounds,
    constraint_bounds=constraint_bounds,
)
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

A function that returns the wrong shape, or a Jacobian without entries, raises a
`ValueError` when the problem compiles. Bounds that do not broadcast to their arrays
raise a `ValueError` when the solve starts, and so does a missing or unexpected
`constraint_bounds`. Ipopt rejecting an option raises a `ValueError` when the solve
starts. If a program fails while Ipopt runs, Ipopt shortens its step, and `solve` raises
a `RuntimeError` with the message of the plugin only when the solve ends without
success.
