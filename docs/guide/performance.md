# Performance

A solve spends its time in three places: the callbacks that evaluate the functions and
their derivatives, the factorization of the linear system inside Ipopt, and the setup of
Ipopt and the linear solver. xipopt shortens the first. The other two are the same as in
any Ipopt interface.

## Comparison with cyipopt

`benchmarks/compare.py` solves a few problems with xipopt and with
[cyipopt](https://github.com/mechmotum/cyipopt), which calls Python functions from the
same Ipopt. The script gives cyipopt the jitted versions of the same JAX functions and
the same sparsity structure, uses MUMPS and one BLAS thread on both sides, and sets the
tolerance to `1e-8`. Both sides then make the same iterations and callbacks. The table
shows the median of 15 solves after a warm-up solve, without the time to compile or jit.

| Problem | Variables | Iterations | cyipopt | xipopt | Ratio |
| --- | --- | --- | --- | --- | --- |
| Problem 71 of Hock and Schittkowski | 4 | 8 | 4.1 ms | 2.5 ms | 1.6x |
| Chained Rosenbrock, equality constraints | 100 | 6 | 8.4 ms | 3.8 ms | 2.2x |
| Chained Rosenbrock, equality constraints | 1000 | 6 | 16.5 ms | 10.5 ms | 1.6x |
| Bratu control | 200 | 2 | 4.1 ms | 2.1 ms | 1.9x |
| Bratu control | 2000 | 2 | 8.6 ms | 6.4 ms | 1.3x |

The machine was a shared virtual machine with four cores of an Intel Xeon at 2.1 GHz, so
treat differences below 10% as noise. The problems converge in a handful of iterations,
which makes the callbacks a large share of the time. The advantage shrinks as the
problem grows, because the factorization takes over. Run the script on your own hardware
and problems to see where yours falls:

```bash
uv pip install --no-build-isolation cyipopt
uv run python benchmarks/compare.py
```

cyipopt has to be built against the Ipopt of the wheel. The workflow
`.github/workflows/benchmark.yml` does that on a GitHub runner.

## Building without line tracing

The development environment compiles the Cython module with line tracing for coverage,
which slows the callbacks down by several times. Reinstall the package without it before
timing anything. The [development page](../development.md) shows how.

## Choosing the linear solver

For problems with thousands of variables the factorization dominates. Compare MUMPS and
SPRAL on your problem with `SolveOptions(linear_solver=...)`, since the better one
depends on the structure of the matrix. SPRAL uses OpenMP threads, so set
`OMP_NUM_THREADS` to share cores sensibly with other work.

## Reducing build time

Compiling five programs takes a second or two for a small problem, and sparsity
detection and compilation grow with the size of the functions. The disk cache removes
that cost for every run after the first. A problem that gives its sparsity patterns
skips detection, which can be most of the build for large functions.
