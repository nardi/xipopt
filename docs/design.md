# Design

This page explains how a solve runs and why it runs quickly. It matters when you profile
a problem or work on the package.

## Programs and callbacks

Ipopt is a C++ library with a C interface. The interface takes five callback functions
when a problem is created, and Ipopt calls them with the current point. xipopt compiles
two programs for them. The first-order program computes the objective, its gradient, the
constraints and the Jacobian in one pass, and the Hessian program computes the Hessian
of the Lagrangian. Ipopt calls the first four callbacks at the same point one after
another, so the first callback at a new point runs the first-order program:

1. The callback copies Ipopt's array `x` into an input buffer that is 64-byte aligned.
2. It calls `execute` from cyxla, which passes the buffers to the PJRT plugin.
3. It stores the four outputs of the program, together with a copy of `x`. A problem
   with state has a fifth output, which the guide on problem state describes.

The other callbacks at the same point copy their values from the stored outputs and run
no program. A callback at another point runs the program again. The point is compared
bit by bit with the stored copy, and Ipopt's `new_x` flag is not used, since that flag
does not reliably show whether the point changed during the restoration phase and after
rejected trial points. A solve clears the stored outputs when it starts, because the
parameters can change between solves.

The input buffers are aliased, so a PJRT buffer on the CPU plugin shares the memory of
an existing array and filling it is a plain copy. The plugin reads the new values at the
next execution. The outputs are read from the plugin's memory in place. A callback
allocates nothing and does not hold the GIL, and the solve itself runs with the GIL
released.

The programs read the variables `x` and the parameters `p`. The Hessian program also
reads the objective factor and the multipliers, which Ipopt passes as arguments to
`eval_h`. The Hessian is a separate program, since Ipopt passes the multipliers to no
other callback.

The Hessian program therefore evaluates the functions at `x` once more, because the
outputs of the first-order program do not contain the intermediate values that second
derivatives need. Inside the Hessian program the functions are evaluated once, since
forward mode over reverse mode reuses that evaluation for all products. With
`Hessian.LIMITED_MEMORY` no Hessian program is compiled, so the functions are evaluated
only once per point.

The first-order program computes derivatives at every new point, including the trial
points that Ipopt evaluates in its line search and then rejects. Those derivatives are
never used, and `unused_derivatives` in the statistics counts them. When the program
fails at a point, or any of its outputs is infinite or not a number, the callbacks at
that point return `false`, so Ipopt treats the point as one where the functions cannot
be evaluated and shortens its step. A gradient that is not a number at a trial point
therefore makes Ipopt reject the point, while an evaluation of the gradient alone at an
accepted point would end the solve.

## Problem state

A `StatefulProblem` adds a vector of state to the programs. The flat parameter input of
the programs contains the parameters followed by the state, so both programs read it
without an additional input, and the derivatives, which are taken with respect to `x`
only, treat it as a constant. The first-order program has a fifth output, the new state,
which the derivative transformations return as an auxiliary value.

After the first-order program succeeds and every output is finite, the callback copies
the new state over the state in the parameter input. A failed execution leaves the state
as it was. The Hessian program has no state output. A solve writes the initial state, or
the state of the solution it warm starts from, into the input before Ipopt starts, and
reads the state back when Ipopt returns. A callback at the stored point runs no program,
so the state does not change either. Because every execution of the first-order program
replaces the state, the results can depend on the order of the evaluations, as the
[guide on problem state](guide/problem-state.md) explains.

The Hessian program reads the state that the first-order program returned at the same
`x`. An inner solve that stops at a tolerance then starts from its solution and needs
one iteration to confirm it, while an inner solve with a fixed number of steps still
takes all of them.

## Callback and factorization time

A callback takes microseconds for a small problem, and its time grows with the size of
the programs. For problems of a few hundred variables the callbacks take most of the
time of a solve, and a Python callback of a jitted function takes far longer per call
than the program itself. For larger problems the factorization of the linear system
takes most of the time, so the choice of linear solver and the sparsity of the
derivatives matter more than the callbacks. `Solution.statistics` shows how the time is
divided for a given problem.

## Failures inside a callback

A callback never raises. If the plugin reports an error, the callback returns `false`,
which makes Ipopt treat the point as one where it cannot evaluate and shorten its step.
The message is stored in the error buffer of cyxla. `solve` raises a `RuntimeError` with
this message when the solve ends without success, and ignores it when Ipopt recovered.

## Callbacks into Python

A solve never runs Python code, except for a problem compiled with
`host_callbacks=True`. A callback in a program calls the native function
`cyxla.python_callback`, which takes the GIL, runs the Python function with copies of
the arguments and copies the results back. The solve thread has released the GIL, so the
native function does not deadlock. An exception becomes the error of that execution, and
the callback returns `false` to Ipopt as for any other failure. The rule that the
callbacks of Ipopt must not raise still applies to the code of xipopt, since the native
function is part of cyxla.

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

Ipopt calls BLAS and LAPACK through libblastrampoline, a library that forwards the calls
to a backend. When the package is imported, it registers the OpenBLAS that it links as
the backend of libblastrampoline. The wheels contain all of these libraries in the
directory of the extension module, and their run paths refer to that directory, so no
system copy is loaded. On Windows the package adds the directory to the search path of
DLLs before the extension is loaded. SPRAL needs OpenMP cancellation, so the package
sets `OMP_CANCELLATION` to `TRUE` before the OpenMP runtime reads its environment.

## Process sharing

One process loads one PJRT plugin, and cyxla owns it. Libraries such as JAX-CIP that
also use cyxla therefore share the client with xipopt.
