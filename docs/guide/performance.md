# Performance

The time of a solve is divided between three parts: the callbacks that evaluate the
functions and their derivatives, the factorization of the linear system inside Ipopt,
and the setup of Ipopt and the linear solver. xipopt reduces the time of the first. The
other two are the same as in any Ipopt interface.

## Comparison with cyipopt

`benchmarks/compare.py` solves a few problems with xipopt and with
[cyipopt](https://github.com/mechmotum/cyipopt), which calls Python functions from the
same Ipopt. The script gives cyipopt the jitted versions of the same JAX functions and
the same sparsity structure, uses MUMPS and one BLAS thread on both sides, and sets the
tolerance to `1e-8`. Both sides then make the same iterations and callbacks. The tables
show the median of 15 solves after a warm-up solve, without the time to compile or jit.

| Problem | Variables | Iterations | cyipopt | xipopt | Ratio |
| --- | --- | --- | --- | --- | --- |
| Problem 71 of Hock and Schittkowski | 4 | 8 | 4.1 ms | 2.5 ms | 1.6x |
| Chained Rosenbrock, equality constraints | 100 | 6 | 8.4 ms | 3.8 ms | 2.2x |
| Chained Rosenbrock, equality constraints | 1000 | 6 | 16.5 ms | 10.5 ms | 1.6x |
| Bratu control | 200 | 2 | 4.1 ms | 2.1 ms | 1.9x |
| Bratu control | 2000 | 2 | 8.6 ms | 6.4 ms | 1.3x |

The same script on a GitHub Actions runner (`benchmark.yml`, Ubuntu with four cores)
gives smaller ratios:

| Problem | Variables | Iterations | cyipopt | xipopt | Ratio |
| --- | --- | --- | --- | --- | --- |
| Problem 71 of Hock and Schittkowski | 4 | 8 | 1.8 ms | 1.2 ms | 1.5x |
| Chained Rosenbrock, equality constraints | 100 | 6 | 2.6 ms | 1.6 ms | 1.6x |
| Chained Rosenbrock, equality constraints | 1000 | 6 | 8.3 ms | 7.0 ms | 1.2x |
| Bratu control | 200 | 2 | 1.3 ms | 1.1 ms | 1.1x |
| Bratu control | 2000 | 2 | 4.4 ms | 4.0 ms | 1.1x |

The first table comes from a shared virtual machine with four cores of an Intel Xeon at
2.1 GHz, so treat differences below 10% as noise. The problems converge in a handful of
iterations, which makes the callbacks a large share of the time. The ratio decreases as
the problem grows, because the factorization takes a larger share of the time. Run the
script on your own hardware and problems to compare:

```bash
uv pip install setuptools
PKG_CONFIG_PATH="$PWD/subprojects/Ipopt-<platform>/lib/pkgconfig" \
    uv pip install --no-build-isolation --no-binary cyipopt cyipopt
uv run python benchmarks/compare.py
```

cyipopt has to be built against the Ipopt of the wheel. The workflow
`.github/workflows/benchmark.yml` does that on a GitHub runner.

## Builds without line tracing

The development environment compiles the Cython module with line tracing for coverage,
which slows the callbacks down by several times. Reinstall the package without it before
timing anything. The [development page](../development.md) shows how.

## Linear solvers

For problems with thousands of variables the factorization dominates. Compare MUMPS and
SPRAL on your problem with `SolveOptions(linear_solver=...)`, since the better one
depends on the structure of the matrix. SPRAL uses OpenMP threads, so set
`OMP_NUM_THREADS` to limit the cores that it uses when other work runs on the machine.

## Build time

Compiling the programs takes a second or two for a small problem, and sparsity detection
and compilation grow with the size of the functions. With the disk cache, every run
after the first skips that work. A problem that gives its sparsity patterns skips
detection, which can be most of the build for large functions.

## Work shared between the objective and the constraints

The objective and the constraints often need the same intermediate result, such as the
solution of an inner solve or a simulation. Separate functions compute it once for each
of them, and the compiler cannot merge the copies when they call code with side effects,
such as a library with native kernels. A single `evaluate` computes it once, and the
first-order program differentiates it once.

<!-- name: test_docs_performance -->
```python
import jax.numpy as jnp
import numpy as np

import xipopt as xip


def evaluate(x, params):
    # This stands for an expensive computation that both parts read.
    state = jnp.sin(x)
    return xip.Evaluation(jnp.sum(state**2), (jnp.sum(state)[None],))


problem = xip.Problem(
    variables=xip.Variable(shape=(2,)),
    evaluate=evaluate,
)
solution = xip.solve(
    problem,
    np.array([0.5, 0.5]),
    variable_bounds=xip.Bounds(-1.0, 1.0),
    constraint_bounds=xip.Bounds(lower=0.1),
)
```

<!-- name: test_docs_performance -->
```python
assert solution.status.succeeded
statistics = solution.statistics
print(statistics.first_order_program, statistics.unused_derivatives)
```

`Solution.statistics` shows where the time of a solve went. The callbacks of Ipopt at
one point share one execution of the first-order program, so its `executions` are fewer
than the sum of the calls of the four callbacks. Ipopt also evaluates trial points that
it rejects, and the program computes the derivatives there too. `unused_derivatives`
counts these points. When it is large compared with the executions and the derivatives
take much longer to compute than the values, the time spent on derivatives at rejected
points is the drawback of this design.

## Loading time of saved problems

Loading a saved problem deserializes one executable per program, which takes most of the
time for large programs. `load` and `loads` deserialize on several threads by default,
since the work releases the GIL, and `workers=1` loads one program after another. A
problem has at most two programs, and the Hessian is usually the larger one, so at best
the loading time is reduced to that of the Hessian program. With a limited-memory
Hessian there is a single program and nothing to run in parallel.

Measurements with `benchmarks/loading.py`, on a problem with an unrolled Newton solve of
40 iterations, on four processors:

| Hessian | Source locations | Programs | File | One thread | Default threads |
|---|---|---:|---:|---:|---:|
| Exact | Kept | 2 | 9.97 MB | 2045 ms | 2339 ms |
| Exact | Removed with `debug_info=False` | 2 | 8.07 MB | 2312 ms | 2220 ms |
| Limited memory | Kept | 1 | 3.29 MB | 882 ms | 905 ms |
| Limited memory | Removed with `debug_info=False` | 1 | 2.72 MB | 788 ms | 717 ms |

The difference between one and several threads is within the variation from run to run
for this problem, so the threads make little difference here. Removing the source
locations reduces the file size by 17 to 19%.

The numbers come from one machine and one problem and vary from run to run. Removing the
source locations also removes the file and line information that profiles of the plugin
use. Passing `warm_up=False` skips the run of each program that follows loading.
