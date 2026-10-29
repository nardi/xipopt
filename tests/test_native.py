"""Tests of the native layer, the layouts, the patterns and the cache."""

import importlib.metadata

import cyxla
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from problems import bratu_control, hs071

import xipopt as xip
from xipopt import _cache, _compile, _layout, _patterns, _solver

EMPTY = np.zeros(0, dtype=np.intc)


def test_ipopt_and_blas() -> None:
    """Checks the linked Ipopt and that OpenBLAS is forwarded to libblastrampoline."""
    assert xip.ipopt_version()[:2] == (3, 14)
    assert _solver.forward_blas() > 0


def test_native_problem_rejects_wrong_programs() -> None:
    """Checks that the number of programs is checked."""
    with pytest.raises(ValueError, match="Expected 5 programs"):
        _solver.NativeProblem(
            [None],
            np.zeros(1),
            np.ones(1),
            np.zeros(0),
            np.zeros(0),
            1,
            EMPTY,
            EMPTY,
            EMPTY,
            EMPTY,
        )


def test_native_problem_rejects_unsupported_shapes() -> None:
    """Checks the errors of Ipopt's problem creation and of the vectors of a solve."""
    with pytest.raises(_solver.SolverError, match="could not create the problem"):
        # A constraint needs at least one Jacobian entry.
        _solver.NativeProblem(
            [None] * 5,
            np.zeros(1),
            np.ones(1),
            np.zeros(1),
            np.ones(1),
            1,
            EMPTY,
            EMPTY,
            EMPTY,
            EMPTY,
        ).solve(np.zeros(1), np.zeros(1))
    compiled = xip.compile(hs071().problem)
    with pytest.raises(ValueError, match="starting point must have shape"):
        compiled.native.solve(np.zeros(3), np.zeros(1))


def test_second_compilation_loads_from_the_cache() -> None:
    """Checks that lowered programs and executables come from the disk cache."""
    problem = bratu_control(7)
    first = xip.compile(problem)
    second = xip.compile(problem)
    assert first.statistics.lowered == 5
    assert first.statistics.compiled == 5
    assert second.statistics.lowered == 0
    assert second.statistics.compiled == 0
    assert second.statistics.loaded == 5
    assert second.statistics.detect_seconds == 0.0
    np.testing.assert_array_equal(
        first.structure.jacobian_rows, second.structure.jacobian_rows
    )
    assert xip.solve(second).objective == pytest.approx(xip.solve(first).objective)


def test_cache_distinguishes_captured_constants() -> None:
    """Checks that two functions differing only in a captured array get different programs."""

    def problem_with(shift: np.ndarray) -> xip.Problem:
        return xip.Problem(
            variables=xip.Variable(shape=(3,)),
            objective=lambda x, params: jnp.sum((x - shift) ** 2),
        )

    first = xip.solve(problem_with(np.array([1.0, 2.0, 3.0])))
    second = xip.solve(problem_with(np.array([3.0, 2.0, 1.0])))
    np.testing.assert_allclose(first.x, [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(second.x, [3.0, 2.0, 1.0], atol=1e-6)


def test_disabled_cache_always_builds() -> None:
    """Checks that a build without the disk cache lowers and compiles every time."""
    options = xip.CompileOptions(disk_cache=False)
    problem = bratu_control(6)
    for _ in range(2):
        statistics = xip.compile(problem, options=options).statistics
        assert statistics.lowered == 5
        assert statistics.loaded == 0


def test_broken_structure_file_is_rebuilt() -> None:
    """Checks that an unreadable structure file is removed and the problem builds again."""
    problem = bratu_control(5)
    before = set(cyxla.cache_directory().glob(f"*{_cache.STRUCTURE_SUFFIX}"))
    xip.compile(problem)
    (file,) = set(cyxla.cache_directory().glob(f"*{_cache.STRUCTURE_SUFFIX}")) - before
    file.write_bytes(b"not an archive")
    assert xip.compile(problem).statistics.lowered == 5
    assert file.read_bytes() != b"not an archive"


def test_key_without_asdex_leaves_it_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that the cache key names only the installed libraries."""

    def missing(name: str) -> None:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    assert _cache.key_distributions() == ("xipopt",)


def test_layouts_reject_wrong_values() -> None:
    """Checks the errors of building and using layouts."""
    with pytest.raises(ValueError, match="lower bound above its upper bound"):
        _layout.variable_layout(xip.Variable(lower=2.0, upper=1.0))
    with pytest.raises(TypeError, match="must be a Variable"):
        _layout.variable_layout({"a": 1.0})
    with pytest.raises(ValueError, match="at least one variable"):
        _layout.variable_layout({})
    layout = _layout.params_layout({"a": jnp.zeros(2)})
    assert layout.size == 2
    assert _layout.params_layout(None).size == 1
    with pytest.raises(ValueError, match="A parameter has shape"):
        layout.flatten({"a": np.zeros(3)})


def test_compile_rejects_malformed_problems() -> None:
    """Checks the errors for objectives, constraints and bounds."""
    variables = xip.Variable(shape=(2,))
    with pytest.raises(ValueError, match="must return a scalar"):
        xip.compile(xip.Problem(variables, objective=lambda x, params: x))
    with pytest.raises(TypeError, match="must be a Constraint"):
        xip.compile(xip.Problem(variables, constraints=(lambda x, p: x,)))  # ty: ignore[invalid-argument-type]
    function = lambda x, params: x  # noqa: E731
    with pytest.raises(ValueError, match="do not broadcast"):
        xip.compile(
            xip.Problem(
                variables, constraints=(xip.Constraint(function, lower=np.zeros(3)),)
            )
        )
    with pytest.raises(ValueError, match="lower bound above its upper bound"):
        xip.compile(
            xip.Problem(
                variables, constraints=(xip.Constraint(function, lower=2.0, upper=1.0),)
            )
        )


def test_output_sizes_are_checked() -> None:
    """Checks that a program with the wrong output size is reported."""

    class Wrong:
        output_bytes = (4,)

    structure = _cache.Structure(EMPTY, EMPTY, EMPTY, EMPTY)
    with pytest.raises(RuntimeError, match="gradient program writes"):
        _compile.check_output_sizes(
            [None, Wrong(), None, None, None],  # ty: ignore[invalid-argument-type]
            3,
            0,
            structure,
        )


def test_patterns_in_every_form() -> None:
    """Checks that all forms of a pattern give the same entries."""
    matrix = np.array([[True, False, True], [False, True, False]])
    expected = _patterns.entries_from_coordinates([0, 0, 1], [0, 2, 1], (2, 3))
    bcoo = sparse.BCOO.fromdense(jnp.asarray(matrix))
    for pattern in (matrix, ([0, 0, 1, 1], [0, 2, 1, 1]), bcoo):
        entries = _patterns.as_entries(pattern, (2, 3))
        np.testing.assert_array_equal(entries.rows, expected.rows)
        np.testing.assert_array_equal(entries.columns, expected.columns)
    lower = _patterns.lower_triangle(_patterns.as_entries(([0, 1], [1, 1]), (2, 2)))
    assert lower.rows.tolist() == [1, 1]
    assert lower.columns.tolist() == [0, 1]
    symmetric = _patterns.symmetric_entries(_patterns.as_entries(([0], [1]), (2, 2)))
    assert symmetric.rows.tolist() == [0, 1]


def test_patterns_are_checked() -> None:
    """Checks the errors for patterns of the wrong shape or with entries outside of it."""
    with pytest.raises(ValueError, match="shape"):
        _patterns.as_entries(np.zeros((2, 2), dtype=bool), (2, 3))
    with pytest.raises(ValueError, match="shape"):
        _patterns.as_entries(sparse.BCOO.fromdense(jnp.eye(2)), (3, 3))
    with pytest.raises(ValueError, match="outside of the shape"):
        _patterns.as_entries(([0, 3], [0, 0]), (2, 2))
    with pytest.raises(ValueError, match="same length"):
        _patterns.as_entries(([0, 1], [0]), (2, 2))

    class Wrong:
        rows = np.array([0])
        cols = np.array([0])
        shape = (5, 5)

    with pytest.raises(ValueError, match="shape"):
        _patterns.as_entries(Wrong(), (2, 2))
