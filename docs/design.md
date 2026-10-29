# Design

This page explains how a solve runs and why it is fast. It matters when you profile a
problem or work on the package.

## Programs and callbacks

Ipopt is a C++ library with a C interface. A user of that interface gives it five
callback functions, and Ipopt calls them with the current point. Each callback of xipopt
runs one program:

1. The callback copies Ipopt's array `x` into an input buffer that is 64-byte aligned.
2. It calls `execute` from cyxla, which passes the buffers to the PJRT plugin.
3. It writes the program's output into the array that Ipopt provides.

The input buffers are aliased. On the CPU plugin a PJRT buffer can share the memory of
an existing array, so filling the buffer is a plain copy and the plugin sees the new
values at the next execution. The outputs are read from the plugin's memory in place. A
callback allocates nothing and does not hold the GIL, and the solve itself runs with the
GIL released.

The programs read the variables `x` and the parameters `p`. The Hessian program also
reads the objective factor and the multipliers, which Ipopt supplies with every call of
`eval_h`.

## Where the time goes

A callback costs microseconds for a small problem, and its time grows with the
size of the programs. For problems of a few hundred variables the callbacks dominate a
solve, and a Python callback of a jitted function costs far more per call than the
program itself. Beyond that the factorization of the linear system takes over, so the
choice of linear solver and the sparsity of the derivatives matter more than the
callbacks. `Solution.statistics` shows how the time splits for a given problem.

## Failures inside a callback

A callback never raises. If the plugin reports an error, the callback returns `false`,
which makes Ipopt treat the point as one where it cannot evaluate and shorten its step.
The message goes to the error buffer of cyxla. `solve` raises it as a `RuntimeError`
when the solve ends without success, and ignores it when Ipopt recovered.

## Lowering and compiling

`compile` traces the functions on flat vectors of variables and parameters, inside
`jax.enable_x64`. It builds the derivative programs with
[asdex](https://pypi.org/project/asdex/) or with dense JAX transformations, lowers all
five programs to StableHLO with `cyxla.lower` for the CPU, and compiles them with
`cyxla.compile_program`. The result is a `NativeProblem`, a Cython object that owns the
buffers and the executables.

The disk cache stores the lowered programs, the sparsity structure and, through cyxla,
the compiled executables.

## Libraries in the wheel

Ipopt calls BLAS and LAPACK through libblastrampoline, a library that forwards them to a
backend. When the package imports, it forwards the OpenBLAS that it links to
libblastrampoline. The wheels bundle all of these libraries next to the extension
module, and their run paths point there, so they never depend on a system copy. On
Windows the package adds the directory to the search path of DLLs before the extension
loads. SPRAL needs OpenMP cancellation, so the package sets `OMP_CANCELLATION` to `TRUE`
before the OpenMP runtime reads its environment.

## Sharing a process

One process loads one PJRT plugin, and cyxla owns it. Libraries such as JAX-CIP that
also use cyxla therefore share the client with xipopt.
