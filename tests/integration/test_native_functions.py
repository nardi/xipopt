"""Tests of problems that call native functions, and of saving them for other processes."""

import importlib
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cyxla
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TESTS_DIRECTORY

import xipopt as xip

PROVIDERS = TESTS_DIRECTORY / "providers"
"""The directory with the stand-in for a library with a native function."""

TARGET = np.array([4.0, 9.0])
"""The values that the squares of the variables approach."""

VARIABLE_BOUNDS = xip.Bounds(0.5, 10.0)
"""The bounds of the variables, which `solve` takes with the problem."""


@pytest.fixture(scope="module")
def square(
    native_library_path: Path, monkeypatch_module: pytest.MonkeyPatch
) -> Callable[..., Any]:
    """Imports the stand-in library, which registers the native function, and returns it."""
    monkeypatch_module.setenv("XIPOPT_NATIVE_LIBRARY", str(native_library_path))
    monkeypatch_module.syspath_prepend(str(PROVIDERS))
    return importlib.import_module("xipopt_native_provider").square


def problem(square: Callable[..., Any]) -> xip.Problem:
    """Returns a problem whose objective squares the variables with the native function."""

    def objective(x: Any, params: Any) -> Any:
        return jnp.sum((square(x) - TARGET) ** 2)

    return xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=objective,
    )


OPTIONS = xip.CompileOptions(
    derivatives=xip.Derivatives.DENSE, disk_cache=False, hessian=xip.Hessian.EXACT
)
"""Dense derivatives, since the sparsity detection does not know calls of FFI targets."""


def test_problem_with_a_native_function_solves(square: Callable[..., Any]) -> None:
    """Checks that Ipopt solves with first and second derivatives through the function."""
    compiled = xip.compile(problem(square), options=OPTIONS)
    solution = xip.solve(
        compiled, np.array([1.0, 1.0]), variable_bounds=VARIABLE_BOUNDS
    )
    assert solution.status is xip.Status.SOLVE_SUCCEEDED
    np.testing.assert_allclose(solution.x, [2.0, 3.0], atol=1e-6)
    assert solution.statistics.hessian.calls > 0


def test_saved_problem_needs_the_library_in_another_process(
    square: Callable[..., Any], native_library_path: Path, tmp_path: Path
) -> None:
    """Checks that a saved problem names its native function, and loads once it is imported."""
    path = tmp_path / "problem.xip"
    xip.save(xip.compile(problem(square), options=OPTIONS), path)
    script = textwrap.dedent(
        f"""
        import sys

        import numpy as np

        import xipopt as xip

        variables = xip.Variable(shape=(2,))
        try:
            xip.load({str(path)!r}, variables)
        except Exception as error:
            assert type(error).__name__ == "MissingTargetError", error
            assert "xipopt_test.square (native function, provided by xipopt_native_provider)" in str(error), error
        else:
            raise AssertionError("The missing function was not reported.")

        sys.path.insert(0, {str(PROVIDERS)!r})
        compiled = xip.load({str(path)!r}, variables, import_providers=True)
        solution = xip.solve(
            compiled,
            np.array([1.0, 1.0]),
            variable_bounds=xip.Bounds(0.5, 10.0),
        )
        assert solution.status is xip.Status.SOLVE_SUCCEEDED, solution.status
        assert abs(solution.x[0] - 2.0) < 1e-6 and abs(solution.x[1] - 3.0) < 1e-6
        assert compiled.statistics.compiled == 0
        """
    )
    environment = {
        **os.environ,
        "XIPOPT_NATIVE_LIBRARY": str(native_library_path),
        cyxla.CACHE_ENVIRONMENT_VARIABLE: str(tmp_path / "cache"),
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    """Returns a `MonkeyPatch` whose changes last for the module."""
    with pytest.MonkeyPatch.context() as patch:
        yield patch
