"""A stand-in for a library with FFI handlers, such as klujax.

Importing it registers the Cholesky kernel of LAPACK under another name. The name is not one
of the targets that JAX promises compatibility for, so programs that call it need the
`custom_calls` option, and a process that loads a saved problem has to import this module.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np

# The module is private to jaxlib, so a type checker does not know it.
_lapack = importlib.import_module("jaxlib.cpu._lapack")

TARGET = "xipopt_test_cholesky"
"""The name under which the handler is registered."""

_lapack.initialize()
jax.ffi.register_ffi_target(
    TARGET, _lapack.registrations()["lapack_dpotrf_ffi"], platform="cpu"
)


@jax.custom_jvp
def cholesky(matrix: jax.Array) -> jax.Array:
    """Returns the lower Cholesky factor of a symmetric positive definite matrix."""
    types = (
        jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    call = jax.ffi.ffi_call(
        TARGET,
        types,
        input_layouts=[(0, 1)],
        output_layouts=[(0, 1), ()],
        input_output_aliases={0: 0},
    )
    factor, _ = call(matrix, uplo=np.uint8(76))
    return jnp.tril(factor)


@cholesky.defjvp
def _cholesky_jvp(primals: tuple[jax.Array], tangents: tuple[jax.Array]):
    """Differentiates through the Cholesky factorization of JAX, which has a rule."""
    (matrix,), (change,) = primals, tangents
    _, derivative = jax.jvp(jnp.linalg.cholesky, (matrix,), (change,))
    return cholesky(matrix), derivative
