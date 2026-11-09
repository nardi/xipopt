# Saved problems

Compiling a problem traces its functions, finds the sparsity of the derivatives, lowers
the programs and compiles them. With the [disk cache](parameters.md), a machine that has
compiled the problem before skips tracing the derivatives, the sparsity detection and
the compilation. Loading a saved file skips all four steps on a machine that matches the
one that saved it. `save` writes a compiled problem to a file, and `load` reads it in
another process, on another machine, or later, without tracing, detecting sparsity or
compiling anything.

## Saved files

<!-- name: test_docs_saving -->
```python
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import xipopt as xip

problem = xip.Problem(
    variables={"position": xip.Variable(shape=(2,))},
    objective=lambda x, params: jnp.sum((x["position"] - params["target"]) ** 2),
    constraints=(lambda x, params: jnp.sum(x["position"]),),
)
params = {"target": np.array([1.0, 1.0])}
compiled = xip.compile(problem, params)

path = Path(tempfile.mkdtemp()) / "problem.xip"
xip.save(compiled, path)
```

The file contains the two programs with their StableHLO, the sparsity structure and the
compile options. It does not contain the bounds, which are passed to `solve`. `.xip` is
the conventional extension. `dumps` returns the same data as bytes, for sending it to a
worker process.

## Loaded problems

The file does not contain the functions of the problem or the trees of its variables and
parameters. `load` takes the trees, so that a loaded problem works like a compiled one:
`x0` and the result use the structure of `variables`, and `solve` takes parameters in
the structure of `params`.

<!-- name: test_docs_saving -->
```python
loaded = xip.load(
    path,
    problem.variables,
    params,
    constraints=(np.zeros(()),),
)
solution = xip.solve(
    loaded,
    params=params,
    variable_bounds=xip.Bounds(-2.0, 2.0),
    constraint_bounds=xip.Bounds(upper=1.0),
)
print(solution.x, solution.objective)
```

<!-- name: test_docs_saving -->
```python
assert loaded.problem is None
assert solution.status.succeeded
assert loaded.statistics.compiled == 0
```

The leaves of `variables` are `Variable`s or arrays, and the leaves of `params` are
arrays or `jax.ShapeDtypeStruct`s, as for `compile`. Only their paths and shapes matter.
They are checked against the file, and a tree with another leaf or shape raises an error
that names the first difference. The bounds do not come from the file. They are passed
to `solve` as `variable_bounds` and `constraint_bounds`. Any pytree that JAX can flatten
works, including named tuples, dataclasses and Equinox modules. The class of a node only
has to exist in the loading process, since nothing about it is stored.

A problem with constraints also needs `constraints`, a tree with the structure and
shapes of `Evaluation.constraints`, as a keyword argument. A problem that was compiled
from a `StatefulProblem` also needs `state`, a tree with the structure and shapes of its
state, as a keyword argument. The initial state itself comes from the file. A problem
without state takes none.

## Other machines

A file contains one executable, compiled for the machine that saved it, together with
the StableHLO of every program. A machine with the same operating system, architecture
and plugin version loads the executable directly. On any other machine the plugin cannot
use it, and `load` compiles the StableHLO instead. This takes about as long as the
compilation of a problem without the lowering and the sparsity detection. Pass
`on_mismatch="raise"` to get a `cyxla.TargetMismatchError` in that case.

The programs always run on the CPU, so a file made on a machine without a GPU loads on a
machine with one and still runs on the CPU.

## Steps a loaded problem skips

Loading does not import asdex or numba, does not call the functions of the problem and
does not trace anything. With `warm_up=True`, which is the default, it runs each program
once, so that the first solve is as fast as the later ones. For a problem with 400
variables a load took about 50 ms, while compiling it without a cache took about 1.5 s.
Importing xipopt and JAX then takes most of the start-up time of a new process, about
0.9 s.

## Trust

A file contains machine code, which the plugin runs when it loads the executable. Load
only files from a source that you trust. The file contains checksums that detect
corruption, but they do not protect against a file that was crafted on purpose.

A compiled problem cannot be pickled, because the trees are not part of the file. Send
the bytes of `dumps` to a worker and call `loads` there with the trees.
