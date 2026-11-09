# Problem state

Some functions contain an inner iteration, such as a Newton solve, whose result at one
point is a good starting guess at the next. A `StatefulProblem` lets the evaluation read
a state from the previous point and return a new one. xipopt stores the state between
evaluations, so the inner iteration starts close to its answer instead of from a fixed
guess.

## A problem with a state

`StatefulProblem` takes the fields of `Problem`, plus the initial `state`. Its
`evaluate` takes the state as a third argument and returns the `Evaluation` together
with the new state. The state is a nonempty pytree of floating point arrays, and the new
state has the same structure and shapes.

<!-- name: test_docs_problem_state -->
```python
import jax.numpy as jnp
import numpy as np

import xipopt as xip


def square_root(target, start):
    root = start
    for _ in range(5):
        root = 0.5 * (root + target / root)
    return root


def evaluate(x, params, state):
    root = square_root(jnp.sum(x**2) + 1.0, state)
    objective = (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2
    return xip.Evaluation(objective, (root,)), root


problem = xip.StatefulProblem(
    variables=xip.Variable(shape=(2,)),
    evaluate=evaluate,
    state=jnp.asarray(1.0),
)
solution = xip.solve(problem, constraint_bounds=xip.Bounds(upper=2.0))
assert solution.status.succeeded
```

The inner iteration above takes five steps from the state. Five steps are enough from a
good guess, but not from the fixed guess `1.0`, which a `Problem` has to use at every
point.

## State updates

Every successful execution of the first-order program replaces the state with the one
that `evaluate` returned. The next execution reads it, whether that is at an accepted
point or at a trial point that Ipopt later rejects. An execution that fails, or that
returns a value that is not finite, leaves the state as it was.

A solve starts from `StatefulProblem.state`. Passing a previous `Solution` as
`warm_start` starts from its `Solution.state` instead.

<!-- name: test_docs_problem_state -->
```python
second = xip.solve(
    problem,
    x0=solution.x,
    warm_start=solution,
    constraint_bounds=xip.Bounds(upper=2.0),
)
assert second.iterations <= solution.iterations
```

The Hessian program reads the state that the last first-order execution returned. It
never changes the state.

## Derivatives

The state is a constant for all derivatives. The gradient, the Jacobian and the Hessian
are taken with respect to `x` only, and no derivative is propagated through the state.
The derivatives therefore have the same meaning as for a `Problem` whose inner iteration
starts from a fixed guess.

## Order-dependent results

The result of `evaluate` depends on which points came before. Keep the answer
independent of the state in exact arithmetic, so the state only chooses where an inner
iteration starts, and run the iteration until it reaches its tolerance. A state that
changes the answer makes the objective a different function after every iteration, and
Ipopt's line search may fail to converge. Use a `Problem` when the evaluation has to be
a pure function of `x` and `params`.

## Saved problems

A saved problem contains its initial state. `load` and `loads` take a tree with the
structure and shapes of the state as the keyword argument `state`, next to `constraints`
for a problem with constraints.

```python
loaded = xip.loads(
    data,
    variables,
    params,
    constraints=constraints,
    state=state,
)
```

A problem without state takes none, and loading a stateful problem without `state`
raises an error.
