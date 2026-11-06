# Libraries with their own kernels

Some JAX libraries call native code through `jax.ffi`, for example `klujax` for sparse
linear solves. Programs that call such a kernel need three things that plain JAX
functions do not.

## Custom call allowance

`jax.export` refuses to serialize a call to a target that it has no compatibility
promise for, and the targets of third-party libraries have none. Compiling a problem
that calls one raises an error that names the option.

```python
options = xip.CompileOptions(custom_calls="detect")
compiled = xip.compile(problem, params, options)
```

`"detect"` lowers each program once more, reads the targets from the result and allows
all of them. A function that is differentiated can call targets that it does not call
itself, such as the transposed solve behind an implicit derivative. A list of names
misses those, so `"detect"` is the safer choice. A tuple of names, such as
`custom_calls=("klujax_solve",)`, allows exactly those targets.

## Sparsity detection

`asdex` cannot trace calls to FFI targets, so `Derivatives.AUTOMATIC` and
`Derivatives.SPARSE` fail for these problems with an error that names the primitive
`ffi_call`. Use `Derivatives.DENSE`, which evaluates the derivatives with JAX, or give
the patterns in the `Problem`.

## Callbacks into Python

`jax.debug.callback`, `jax.pure_callback` and `jax.experimental.io_callback` call back
into Python, which a program cannot do on its own. Compiling a problem with one raises
an error that lists each callback and its source line.

`equinox.error_if` is an example of a common function that works via a callback. With
the default `EQX_ON_ERROR=raise` it adds a callback to the program. Set
`EQX_ON_ERROR=nan` before Equinox is imported, and a failed check returns NaN instead.
Ipopt shortens the step when the objective or a constraint is not a number, so the check
does what it was written for, and the solve never enters Python.

```python
import os

os.environ["EQX_ON_ERROR"] = "nan"

import equinox as eqx  # noqa: E402
```

`CompileOptions(host_callbacks=True)` keeps the callbacks and runs them inside the
programs instead. A callback that raises ends that evaluation with an error. Ipopt
treats this like a point where the functions cannot be evaluated and shortens the step.
If it happens at every point, the solve fails with the message of the exception.

```python
options = xip.CompileOptions(host_callbacks=True, derivatives=xip.Derivatives.DENSE)
compiled = xip.compile(problem, params, options)
```

With the option, each callback takes the GIL, so a solve is slower for every callback
it makes. `asdex` cannot trace callbacks, so the derivatives must be dense or
come with patterns. The programs refer to callbacks of the running process, so such a
problem cannot be saved and does not use the disk cache. Ordered callbacks are not
supported.

## Native functions

A C, C++, Rust or Cython kernel can be offered to xipopt problems as a native function
of cyxla. The function receives the buffers of the arguments and writes its results in
place, without the GIL, and a problem that calls it solves without entering Python. The
[native functions guide of cyxla](https://nardi.github.io/cyxla/guide/native-functions/)
describes the C interface and the Cython way to register one.

A function has no derivative by itself, so a library offers it as a primitive from
`cyxla.native_primitive`, with a `jvp` rule written in `jnp` operations. Then Ipopt gets
first derivatives from the rule and exact second derivatives from the Hessian program. A
`custom_vjp` function gives no exact Hessian, because forward mode cannot differentiate
it.

```python
# Importing the library registers the native function and offers the primitive `square`.
import mykernels

# The bounds go to `xip.solve` as `variable_bounds=xip.Bounds(0.5, 10.0)`.
problem = xip.Problem(
    variables=xip.Variable(shape=(2,)),
    evaluate=xip.evaluation_from_functions(
        objective=lambda x, params: jnp.sum((mykernels.square(x) - target) ** 2)
    ),
)
options = xip.CompileOptions(derivatives=xip.Derivatives.DENSE)
```

Native functions need `Derivatives.DENSE`, or patterns that the problem gives, for the
same reason as other FFI calls. They need no `custom_calls` option. A saved problem
records the name of each function with the package and version that registered it. The
process that loads the file imports the library first, or passes
`import_providers=True`.

## Saved problems

A saved problem refers to a kernel by name, and the kernel is in the library that
registered it. The process that loads the file needs the library imported first, as it
would to compile the problem. Loading checks this and raises a
`cyxla.MissingTargetError` that names each missing target with the package and the
version that provided it when the file was made.

```python
import klujax  # noqa: F401

loaded = xip.load("problem.xip", problem.variables, params)
```

`xip.load(..., import_providers=True)` imports those packages itself. Importing a
package by name runs its code, so use the option only for files you trust. The provider
is recorded when the library registered its kernels after `cyxla` was imported. A
library that was imported earlier leaves the provider unknown, and the error then names
the target only.

## Warm-up

`load` runs every program once to page in its code. The run uses the point nearest to
the origin inside the `variable_bounds` given to `load`, which is zero without bounds,
and the parameters given to `load`. Parameters that are only `jax.ShapeDtypeStruct`s run
as zeros. A program that fails on that input, for instance because it factorizes a
singular matrix, raises a `RuntimeWarning` and loading continues. Pass `warm_up=False`
to skip the run.
