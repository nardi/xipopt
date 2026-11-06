# Design

This page explains how a solve runs and why it runs quickly. It matters when you profile
a problem or work on the package.

## Programs and callbacks

Ipopt is a C++ library with a C interface. A user of that interface gives it five
callback functions, and Ipopt calls them with the current point. xipopt compiles two
programs for them. The first-order program computes the objective, its gradient, the
constraints and the Jacobian in one pass, and the Hessian program computes the Hessian
of the Lagrangian. Ipopt calls the first four at the same point one after another, so
the first callback at a new point runs the first-order program:

1. The callback copies Ipopt's array `x` into an input buffer that is 64-byte aligned.
2. It calls `execute` from cyxla, which passes the buffers to the PJRT plugin.
3. It keeps the four outputs of the program, together with a copy of `x`.

The other callbacks at the same point copy their share from the kept outputs and run no
program. A callback at another point runs the program again. The point is compared bit
by bit with the kept one, and not through Ipopt's `new_x` flag, since the restoration
phase and rejected trial points make that flag a poor guide to whether the point
changed. A solve clears the kept outputs when it starts, because the parameters can
change between solves.

The input buffers are aliased. On the CPU plugin a PJRT buffer can share the memory of
an existing array, so filling the buffer is a plain copy and the plugin sees the new
values at the next execution. The outputs are read from the plugin's memory in place. A
callback allocates nothing and does not hold the GIL, and the solve itself runs with the
GIL released.

The programs read the variables `x` and the parameters `p`. The Hessian program also
reads the objective factor and the multipliers, which Ipopt supplies with every call of
`eval_h`. The Hessian cannot join the first-order program, since Ipopt supplies the
multipliers only with the call of `eval_h`.

The Hessian program therefore evaluates the functions at `x` once more, because the
outputs of the first-order program do not contain the intermediate values that second
derivatives need. Inside the Hessian program the evaluation runs once, since forward
mode over reverse mode shares it. `Hessian.LIMITED_MEMORY` leaves out the Hessian
program and with it this second evaluation.

The first-order program computes derivatives at every new point, including the trial
points that Ipopt evaluates in its line search and then rejects. Those derivatives are
wasted, and `unused_derivatives` in the statistics counts them. When the program fails
at a point, or any of its outputs is infinite or not a number, the callback that came
first reports the point as one where the functions cannot be evaluated, and Ipopt
shortens its step. A gradient that is not a number at a trial point therefore rejects
the point, where an evaluation of the gradient alone at an accepted point would end the
solve.

## Callback and factorization time

A callback costs microseconds for a small problem, and its time grows with the
size of the programs. For problems of a few hundred variables the callbacks dominate a
solve, and a Python callback of a jitted function costs far more per call than the
program itself. Beyond that the factorization of the linear system dominates, so the
choice of linear solver and the sparsity of the derivatives matter more than the
callbacks. `Solution.statistics` shows how the time splits for a given problem.

## Failures inside a callback

A callback never raises. If the plugin reports an error, the callback returns `false`,
which makes Ipopt treat the point as one where it cannot evaluate and shorten its step.
The message goes to the error buffer of cyxla. `solve` raises it as a `RuntimeError`
when the solve ends without success, and ignores it when Ipopt recovered.

## Callbacks into Python

The promise above, that a solve never enters the interpreter, applies unless the problem
is compiled with `host_callbacks=True`. A callback in a program calls the native
function `cyxla.python_callback`, which takes the GIL, runs the Python function with
copies of the arguments and copies the results back. The solve thread has released the
GIL, so the native function does not deadlock. An exception becomes the error of that
execution, and the callback returns `false` to Ipopt as for any other failure. The rule
that the callbacks of Ipopt must not raise still applies to the code of xipopt, since
the native function belongs to cyxla.

## Lowering and compilation

`compile` traces the functions on flat vectors of variables and parameters, inside
`jax.enable_x64`. It builds the derivative programs with
[asdex](https://pypi.org/project/asdex/) or with dense JAX transformations, lowers all
the programs to StableHLO with `cyxla.lower` for the CPU, and compiles them with
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

## Process sharing

One process loads one PJRT plugin, and cyxla owns it. Libraries such as JAX-CIP that
also use cyxla therefore share the client with xipopt.
