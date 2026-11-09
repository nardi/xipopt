# Options and callbacks

[`SolveOptions`][xipopt.SolveOptions] holds the settings of one solve. They apply to a
single call, so a compiled problem can be solved with different options without
compiling again.

## Common options

<!-- name: test_docs_options -->
```python
import jax.numpy as jnp
import numpy as np

import xipopt as xip

problem = xip.Problem(
    variables=xip.Variable(shape=(2,)),
    objective=lambda x, params: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
)
compiled = xip.compile(problem)
start = np.array([-1.2, 1.0])
variable_bounds = xip.Bounds(-2.0, 2.0)

options = xip.SolveOptions(
    tolerance=1e-10,
    max_iterations=200,
    time_limit=10.0,
    linear_solver=xip.LinearSolver.SPRAL,
)
solution = xip.solve(compiled, start, options=options, variable_bounds=variable_bounds)
print(solution.status.name, solution.iterations)
```

<!-- name: test_docs_options -->
```python
assert solution.status.succeeded
```

- `tolerance` is Ipopt's `tol`, the tolerance on the scaled optimality error.
- `max_iterations` and `time_limit` stop a solve. The time limit counts seconds of wall
  time, and an infinite limit means none.
- `linear_solver` is `MUMPS` or `SPRAL`. The wheels contain these two. MUMPS is the
  default. SPRAL runs in parallel with OpenMP, so it can be faster for large problems.
- `print_level` is Ipopt's, from 0 to 12. Ipopt prints through the C library, so the
  output does not go through Python's `sys.stdout`. The default 0 prints nothing.
- `blas_threads` sets the number of OpenBLAS threads, one by default. The linear algebra
  calls of a sparse factorization are small, and a single thread avoids the overhead of
  synchronizing many. Pass `None` to skip setting the thread count.

Any other option of Ipopt is passed in `settings`, as a name and a value of type `str`,
`int` or `float`. Ipopt's [option list](https://coin-or.github.io/Ipopt/OPTIONS.html)
has them all. The settings are applied after the fields above, so they override them.

<!-- name: test_docs_options -->
```python
options = xip.SolveOptions(
    settings=(
        xip.IpoptSetting("mu_strategy", "adaptive"),
        xip.IpoptSetting("acceptable_tol", 1e-6),
    )
)
solution = xip.solve(compiled, start, options=options, variable_bounds=variable_bounds)
```

<!-- name: test_docs_options -->
```python
assert solution.status.succeeded
```

Ipopt rejects an unknown name or a value that does not fit, and `solve` then raises a
`ValueError` that names the option.

## Iteration callback

`SolveOptions(callback=...)` takes a function that Ipopt calls at the end of every
iteration with an [`IterationInfo`][xipopt.IterationInfo]. It contains the iteration,
the objective, the infeasibilities, the barrier parameter, the step sizes and the
current point in the structure of the variables. Returning `False` stops the solve with
`Status.USER_REQUESTED_STOP`, and returning nothing continues. An exception ends the
solve and is raised from `solve`.

<!-- name: test_docs_options -->
```python
history = []


def watch(info):
    history.append((info.iteration, info.objective, info.primal_infeasibility))
    return info.iteration < 5


solution = xip.solve(
    compiled,
    start,
    options=xip.SolveOptions(callback=watch, tolerance=1e-12),
    variable_bounds=variable_bounds,
)
```

<!-- name: test_docs_options -->
```python
assert solution.status is xip.Status.USER_REQUESTED_STOP
assert history[-1][0] == 5
```

The callback runs with the GIL, so it is the one place where a solve runs Python code.
Without a callback Ipopt only counts the iterations.

## Statistics

A [`Solution`][xipopt.Solution] contains the number of callbacks that Ipopt made and of
executions of the programs. `statistics` has, for each of the five callbacks, the number
of calls and how many of them used the outputs of an earlier execution at the same
point. It also has the executions and the seconds of the first-order program and of the
Hessian program, and the number of points at which derivatives were computed and never
used.

<!-- name: test_docs_options -->
```python
solution = xip.solve(compiled, start, variable_bounds=variable_bounds)
print(solution.statistics.gradient, solution.statistics.hessian)
```

<!-- name: test_docs_options -->
```python
assert solution.statistics.objective.calls >= solution.iterations
```

## Threads

A solve holds a lock for the length of the run, because the runtime of cyxla and the
buffers that Ipopt's callbacks share are not thread-safe. Solves from several threads
therefore run one after another. A solve releases the GIL while Ipopt works, so other
Python threads keep running.
