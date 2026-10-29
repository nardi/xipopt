# Parameters and warm starts

The functions of a problem may depend on data that changes between solves, such as the
prices in a portfolio or the target of a tracking problem. Passing the data as
parameters keeps the compiled programs valid for all of it.

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
    variables=xip.Variable(lower=-2.0, upper=2.0, shape=(3,)),
    objective=objective,
    constraints=(xip.Constraint(lambda x, params: jnp.sum(x), upper=1.0),),
)
layout = {
    "target": jax.ShapeDtypeStruct((3,), jnp.float64),
    "weight": jax.ShapeDtypeStruct((), jnp.float64),
}
compiled = xip.compile(problem, layout)

for weight in (0.0, 1.0, 10.0):
    params = {"target": np.array([1.0, 0.5, 0.0]), "weight": weight}
    solution = xip.solve(compiled, params=params)
    print(weight, solution.objective)
```

<!-- name: test_docs_parameters -->
```python
assert solution.status.succeeded
```

The bounds of variables and constraints are fixed when the problem compiles. Data that
moves a bound goes into the function instead, for instance as `x - params["limit"]` with
an upper bound of zero.

The programs hold all values as `float64`. A parameter of another type, such as an
integer count, is converted to `float64` on the way in and back to its own type inside
the function, so integers up to 2^53 survive exactly.

## Warm starts

Parameters that change a little between solves give solutions that are close to each
other. Starting the next solve from the previous point and its multipliers saves
iterations. Pass the earlier [`Solution`][xipopt.Solution] as `warm_start` and its point
as the start:

<!-- name: test_docs_parameters -->
```python
params = {"target": np.array([1.0, 0.5, 0.0]), "weight": 1.0}
cold = xip.solve(compiled, params=params)

params["target"] = np.array([1.05, 0.5, 0.0])
warm = xip.solve(
    compiled,
    cold.x,
    params,
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
so the settings above make the push small. The values suit a small change of the
parameters and are a starting point for tuning, as [Ipopt's
documentation](https://coin-or.github.io/Ipopt/OPTIONS.html) describes.

## The disk cache

Lowering and compiling dominate the time of a first solve. Both are stored in the
directory of the cyxla cache, which is `~/.cache/cyxla` unless `CYXLA_CACHE_DIR` says
otherwise. The key covers the traced functions, including the values of the constants
they capture, the sparsity patterns that the problem gives, the compile options, the
versions of JAX, cyxla, xipopt and asdex. Executables also depend on the loaded plugin.
A second process that builds the same problem skips tracing the derivatives, the
sparsity detection and the compilation.

`CompileOptions(disk_cache=False)` turns the cache off. The `build` field of a
[`Solution`][xipopt.Solution] and the `statistics` of a compiled problem count the
programs that were lowered, compiled and loaded.
