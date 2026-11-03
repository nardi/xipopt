"""A stand-in for a library that offers a native function, such as a compiled kernel package.

Importing it registers the function `xipopt_test.square` from the shared library whose path is
in `XIPOPT_NATIVE_LIBRARY`, and offers it as the primitive `square`, which has a derivative.
"""

import ctypes
import os
from collections.abc import Callable, Sequence
from typing import Any

import cyxla

LIBRARY = ctypes.CDLL(os.environ["XIPOPT_NATIVE_LIBRARY"])
"""The shared library with the C function."""


class SquareKernel:
    """The native squaring function as a kernel, with a derivative written with `jnp`."""

    name = "xipopt_test.square"
    user_data_address = 0

    def __init__(self) -> None:
        self.function_address = ctypes.cast(LIBRARY.square, ctypes.c_void_p).value or 0
        self.square: Callable[..., Any] | None = None

    def result_avals(self, x: Any) -> Sequence[Any]:
        """The result has the type of the argument."""
        return (x,)

    def jvp(self, primals: Any, tangents: Any) -> tuple[list[Any], list[Any]]:
        """Differentiates with `jnp`, so a solver can get second derivatives."""
        assert self.square is not None
        (x,), (tangent,) = primals, tangents
        return [self.square(x)], [2.0 * x * tangent]


_kernel = SquareKernel()
square = _kernel.square = cyxla.native_primitive(_kernel)
"""The primitive that squares an array with the native function."""
