# Parameters and warm starts

The functions of a problem may depend on data that changes between solves, such as the
prices in a portfolio or the target of a tracking problem. When the data is passed as
parameters, one compiled problem can be solved for all of its values.

## Parameters

`compile` and `solve` take a pytree `params`, and every function receives it as its
second argument. Compiling needs only the structure, shapes and types of the parameters,
so it accepts arrays or `jax.ShapeDtypeStruct`s. Solving takes any values of the same
layout.

<!-- name: test_docs_parameters -->
```python
import jax
import jax.numpy as jnp
import numpy as np

import xipopt as xip


def objective(x, params):
    return jnp.sum((x - params["target"]) ** 2) + params["weight"] * jnp.sum(x**4)


problem = xip.Problem(
    variables=xip.Variable(shape=(3,)),
    objective=objective,
    constraints=(lambda x, params: jnp.sum(x),),
)
variable_bounds = xip.Bounds(-2.0, 2.0)
constraint_bounds = xip.Bounds(upper=1.0)
layout = {
    "target": jax.ShapeDtypeStruct((3,), jnp.float64),
    "weight": jax.ShapeDtypeStruct((), jnp.float64),
}
compiled = xip.compile(problem, layout)

for weight in (0.0, 1.0, 10.0):
    params = {"target": np.array([1.0, 0.5, 0.0]), "weight": weight}
    solution = xip.solve(
        compiled,
        params=params,
        variable_bounds=variable_bounds,
        constraint_bounds=constraint_bounds,
    )
    print(weight, solution.objective)
```

<!-- name: test_docs_parameters -->
```python
assert solution.status.succeeded
```

The bounds are not part of the compiled problem. They are passed to `solve` on each
call, so they can change between solves as the parameters do. A bound that depends on
data needs no rewrite of the function.

The programs represent all values as `float64`. A parameter of another type, such as an
integer count, is converted to `float64` in the input of the program and back to its own
type inside the function, so integers up to 2^53 are represented exactly.

## Warm starts

Parameters that change slightly between solves give solutions that are close to each
other. Starting the next solve from the previous point and its multipliers reduces the
number of iterations. Pass the earlier [`Solution`][xipopt.Solution] as `warm_start` and
its point as the start:

<!-- name: test_docs_parameters -->
```python
params = {"target": np.array([1.0, 0.5, 0.0]), "weight": 1.0}
cold = xip.solve(
    compiled,
    params=params,
    variable_bounds=variable_bounds,
    constraint_bounds=constraint_bounds,
)

params["target"] = np.array([1.05, 0.5, 0.0])
warm = xip.solve(
    compiled,
    cold.x,
    params,
    variable_bounds=variable_bounds,
    constraint_bounds=constraint_bounds,
    options=xip.SolveOptions(
        settings=(
            xip.IpoptSetting("mu_init", 1e-6),
            xip.IpoptSetting("warm_start_bound_push", 1e-8),
            xip.IpoptSetting("warm_start_mult_bound_push", 1e-8),
        )
    ),
    warm_start=cold,
)
print(cold.iterations, warm.iterations)
```

<!-- name: test_docs_parameters -->
```python
assert warm.status.succeeded
```

A warm start sets Ipopt's `warm_start_init_point` to `yes`. Ipopt pushes the starting
point away from its bounds by default, which undoes most of the benefit of a good start,
so the settings above set the push to a small value. The values suit a small change of
the parameters and are a starting point for tuning, as [Ipopt's
documentation](https://coin-or.github.io/Ipopt/OPTIONS.html) describes.

## The disk cache

Lowering and compiling take most of the time of a first solve. The results of both are
stored in the directory of the cyxla cache, which is `~/.cache/cyxla` unless
`CYXLA_CACHE_DIR` is set. The key includes the traced functions with the values of the
constants they capture, the sparsity patterns that the problem gives, the compile
options and the versions of JAX, cyxla, xipopt and asdex. The key of an executable also
includes the loaded plugin. A second process that builds the same problem skips tracing
the derivatives, the sparsity detection and the compilation.

`CompileOptions(disk_cache=False)` turns the cache off. The `build` field of a
[`Solution`][xipopt.Solution] and the `statistics` of a compiled problem contain the
number of programs that were lowered, compiled and loaded.
