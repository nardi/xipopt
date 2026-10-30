"""Tests that a fresh process loads and solves a saved problem."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from problems import hs071

import xipopt as xip


def test_another_process_loads_without_the_problem(tmp_path: Path) -> None:
    """Checks that a fresh process solves a saved problem without detection or tracing."""
    known = hs071()
    path = tmp_path / "problem.xip"
    xip.save(xip.compile(known.problem), path)
    script = textwrap.dedent(
        f"""
        import sys

        import numpy as np

        import xipopt as xip

        variables = xip.Variable(lower=1.0, upper=5.0, shape=(4,))
        compiled = xip.load({str(path)!r}, variables)
        solution = xip.solve(compiled, np.array([1.0, 5.0, 5.0, 1.0]))
        assert solution.status is xip.Status.SOLVE_SUCCEEDED, solution.status
        assert abs(solution.objective - 17.0140173) < 1e-6
        for module in ("asdex", "numba", "problems"):
            assert module not in sys.modules, module
        assert compiled.statistics.compiled == 0
        """
    )
    environment = {"CYXLA_CACHE_DIR": str(tmp_path / "empty-cache")}

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
