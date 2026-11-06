"""Tests of problems that call the handlers of a library, saved and loaded in other processes."""

import importlib
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import cyxla
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TESTS_DIRECTORY

import xipopt as xip

PROVIDERS = TESTS_DIRECTORY / "providers"
"""The directory with the stand-in for a library with FFI handlers."""

OPTIMUM = 4.5
"""The objective of `problem` at its minimum, which lies at `[0.5, 0]`."""

DENSE = xip.CompileOptions(derivatives=xip.Derivatives.DENSE)
"""Dense derivatives, since the sparsity detection does not know calls to FFI targets."""


@pytest.fixture(scope="module")
def provider() -> Iterator[ModuleType]:
    """Imports the stand-in library, which registers its handler as a side effect."""
    sys.path.insert(0, str(PROVIDERS))
    try:
        yield importlib.import_module("xipopt_test_provider")
    finally:
        sys.path.remove(str(PROVIDERS))


def problem(provider: ModuleType) -> xip.Problem:
    """Returns a problem whose objective factorizes a matrix with the handler of the library."""

    def objective(x: Any, params: Any) -> Any:
        matrix = jnp.array(
            [[x[0] ** 2 + 2.0, x[0] * x[1]], [x[0] * x[1], x[1] ** 2 + 2.0]]
        )
        return jnp.sum(provider.cholesky(matrix) ** 2) + (x[0] - 1.0) ** 2

    return xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=objective,
    )


def test_custom_calls_have_to_be_allowed(provider: ModuleType) -> None:
    """Checks that the error names the option that allows the targets of a library."""
    with pytest.raises(ValueError, match=r'CompileOptions\(custom_calls="detect"\)'):
        xip.compile(problem(provider), options=DENSE._replace(disk_cache=False))


@pytest.mark.parametrize("custom_calls", ["detect", ("xipopt_test_cholesky",)])
def test_allowed_custom_calls_solve(
    provider: ModuleType, custom_calls: tuple[str, ...] | Literal["detect"]
) -> None:
    """Checks that a problem solves once its targets are allowed by name or by detection."""
    options = DENSE._replace(custom_calls=custom_calls, disk_cache=False)
    compiled = xip.compile(problem(provider), options=options)
    solution = xip.solve(compiled)
    assert solution.status is xip.Status.SOLVE_SUCCEEDED
    assert solution.objective == pytest.approx(OPTIMUM, abs=1e-8)
    np.testing.assert_allclose(solution.x, [0.5, 0.0], atol=1e-6)


def run_loader(
    path: Path, tmp_path: Path, arguments: str, python_path: Path | None
) -> subprocess.CompletedProcess[str]:
    """Loads a saved problem in a fresh process, and solves it unless that fails."""
    script = textwrap.dedent(
        f"""
        import xipopt as xip

        variables = xip.Variable(shape=(2,))
        compiled = xip.load({str(path)!r}, variables{arguments})
        solution = xip.solve(compiled)
        assert solution.status is xip.Status.SOLVE_SUCCEEDED, solution.status
        assert abs(solution.objective - {OPTIMUM}) < 1e-8, solution.objective
        """
    )
    environment = {
        **os.environ,
        cyxla.CACHE_ENVIRONMENT_VARIABLE: str(tmp_path / "cache"),
    }
    if python_path is None:
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(python_path)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_loading_needs_the_library(provider: ModuleType, tmp_path: Path) -> None:
    """Checks the error of a process that has not imported the library of a saved problem."""
    options = DENSE._replace(custom_calls="detect", disk_cache=False)
    path = tmp_path / "problem.xip"
    xip.save(xip.compile(problem(provider), options=options), path)

    result = run_loader(path, tmp_path, "", None)
    assert result.returncode != 0
    assert "MissingTargetError" in result.stderr
    assert (
        "xipopt_test_cholesky (FFI target, provided by xipopt_test_provider"
        in result.stderr
    )

    # The library comes with the process this time, so loading finds its handler.
    script_with_import = run_loader(
        path, tmp_path, ", import_providers=True", PROVIDERS
    )
    assert script_with_import.returncode == 0, script_with_import.stderr
    refused = run_loader(path, tmp_path, ", import_providers=False", PROVIDERS)
    assert refused.returncode != 0
    assert "MissingTargetError" in refused.stderr
