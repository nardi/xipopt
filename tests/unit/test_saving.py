"""Tests of saving a compiled problem and loading it in the same and in another process."""

import dataclasses
import json
import math
import warnings
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import cyxla
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import BRATU_BOUNDS, bratu_control, hs071

import xipopt as xip
from xipopt import _bounds, _serialize
from xipopt._layout import leaf_shape, params_layout, tree_layout, variable_layout
from xipopt._types import Program

HS071_CONSTRAINTS = (np.zeros(()), np.zeros(()))
"""The constraints of problem 71 as a tree of arrays, which `load` checks the file against."""


def solved(
    compiled: xip.CompiledProblem,
    start: Any = None,
    *,
    variable_bounds: xip.Bounds | None = None,
    constraint_bounds: xip.Bounds | None = None,
) -> xip.Solution:
    """Solves a compiled problem with a tight tolerance and the given bounds."""
    return xip.solve(
        compiled,
        start,
        options=xip.SolveOptions(tolerance=1e-10),
        variable_bounds=variable_bounds,
        constraint_bounds=constraint_bounds,
    )


def test_loaded_problem_solves_like_the_original(tmp_path: Path) -> None:
    """Checks that a problem saved to a file gives the same solution after loading."""
    known = hs071()
    compiled = xip.compile(known.problem)
    xip.save(compiled, tmp_path / "hs071.xip")
    loaded = xip.load(
        tmp_path / "hs071.xip", known.problem.variables, constraints=HS071_CONSTRAINTS
    )
    assert loaded.problem is None
    assert loaded.sense is compiled.sense
    assert loaded.options == compiled.options
    assert loaded.constraints.shapes == compiled.constraints.shapes
    assert loaded.statistics.loaded == 2
    assert loaded.statistics.compiled == 0
    first = solved(
        compiled,
        known.start,
        variable_bounds=known.variable_bounds,
        constraint_bounds=known.constraint_bounds,
    )
    second = solved(
        loaded,
        known.start,
        variable_bounds=known.variable_bounds,
        constraint_bounds=known.constraint_bounds,
    )
    assert second.status is first.status
    assert second.objective == pytest.approx(first.objective, abs=1e-12)
    np.testing.assert_allclose(second.x, first.x, atol=1e-10)
    assert second.iterations == first.iterations


def test_bytes_round_trip_with_pytrees_and_parameters() -> None:
    """Checks a problem with dictionary variables and parameters, loaded from bytes."""
    base = bratu_control(10)
    evaluate = base.evaluate
    params = {"scale": jax.ShapeDtypeStruct((), jnp.float64)}

    def scaled(x: Any, p: Any) -> xip.Evaluation:
        result = evaluate(x, p)
        return xip.Evaluation(p["scale"] * result.objective, result.constraints)

    problem = dataclasses.replace(base, evaluate=scaled)
    compiled = xip.compile(problem, params)
    data = xip.dumps(compiled)
    template = {
        "u": xip.Variable(shape=(10,)),
        "c": xip.Variable(shape=(10,)),
    }
    loaded = xip.loads(
        data, template, {"scale": np.float64(1.0)}, constraints=np.zeros(10)
    )
    assert [leaf.path for leaf in loaded.variables.fingerprint] == ["['c']", "['u']"]
    original = xip.solve(
        compiled, params={"scale": 2.0}, constraint_bounds=BRATU_BOUNDS
    )
    again = xip.solve(loaded, params={"scale": 2.0}, constraint_bounds=BRATU_BOUNDS)
    assert again.objective == pytest.approx(original.objective, rel=1e-9)
    assert set(again.x) == {"u", "c"}
    np.testing.assert_allclose(again.x["u"], original.x["u"], atol=1e-8)


def test_arrays_can_stand_in_for_variables() -> None:
    """Checks that a template of arrays works, and that the bounds of a solve apply to a loaded problem."""
    problem = xip.Problem(
        variables={"a": xip.Variable(shape=(2,))},
        evaluate=xip.evaluation_from_functions(objective=lambda x, p: jnp.sum(x["a"])),
    )
    data = xip.dumps(xip.compile(problem))
    loaded = xip.loads(data, {"a": np.zeros(2)}, warm_up=False)
    solution = xip.solve(loaded, variable_bounds=xip.Bounds(lower=1.0, upper=2.0))
    np.testing.assert_allclose(solution.x["a"], 1.0, atol=1e-6)


class Plant(NamedTuple):
    """A named tuple of variables."""

    flow: Any
    count: Any


@dataclasses.dataclass
class Settings:
    """A plain dataclass of parameters."""

    weight: Any
    offset: Any


jax.tree_util.register_dataclass(Settings, ["weight", "offset"], [])


def test_named_tuples_and_dataclasses() -> None:
    """Checks that other pytree types come back as the caller's own types."""
    variables = Plant(xip.Variable(), xip.Variable(shape=(2,)))
    problem = xip.Problem(
        variables=variables,
        evaluate=xip.evaluation_from_functions(
            objective=lambda x, p: (
                (x.flow - p.offset) ** 2 + p.weight * jnp.sum(x.count**2)
            )
        ),
    )
    params = Settings(np.float64(1.0), np.float64(0.0))
    loaded = xip.loads(
        xip.dumps(xip.compile(problem, params)),
        Plant(np.zeros(()), np.zeros(2)),
        Settings(jax.ShapeDtypeStruct((), jnp.float64), np.float64(0.0)),
    )
    solution = xip.solve(
        loaded,
        params=Settings(1.0, 2.0),
        variable_bounds=xip.Bounds(
            lower=Plant(-3.0, -math.inf), upper=Plant(3.0, math.inf)
        ),
    )
    assert isinstance(solution.x, Plant)
    assert solution.x.flow == pytest.approx(2.0, abs=1e-6)


def test_trees_are_checked_against_the_file() -> None:
    """Checks the errors for variables and parameters that do not match."""
    problem = xip.Problem(
        variables={"a": xip.Variable(shape=(2,)), "b": xip.Variable()},
        evaluate=xip.evaluation_from_functions(
            objective=lambda x, p: jnp.sum(x["a"]) ** 2 + x["b"] ** 2 + jnp.sum(p["w"])
        ),
    )
    data = xip.dumps(xip.compile(problem, {"w": np.ones(3)}))
    good = {"a": np.zeros(2), "b": np.zeros(())}
    with pytest.raises(ValueError, match="variables have the leaf"):
        xip.loads(data, {"a": np.zeros(2), "c": np.zeros(())}, {"w": np.ones(3)})
    with pytest.raises(ValueError, match="variables have the shape"):
        xip.loads(data, {"a": np.zeros(3), "b": np.zeros(())}, {"w": np.ones(3)})
    with pytest.raises(ValueError, match="parameters have the shape"):
        xip.loads(data, good, {"w": np.ones(4)})
    with pytest.raises(ValueError, match="parameters have the leaf"):
        xip.loads(data, good, {"v": np.ones(3)})
    assert xip.loads(data, good, {"w": np.ones(3)}).params.size == 3


def test_a_file_from_another_machine_is_compiled_again() -> None:
    """Checks the fallback for a fingerprint that the file has no executable for."""
    known = hs071()
    compiled = xip.compile(known.problem)
    data = xip.dumps(compiled)
    other = cyxla.target_fingerprint()._replace(machine="other")
    with cyxla.override_target(other):
        with pytest.raises(cyxla.TargetMismatchError):
            xip.loads(
                data,
                known.problem.variables,
                constraints=HS071_CONSTRAINTS,
                on_mismatch="raise",
            )
        loaded = xip.loads(data, known.problem.variables, constraints=HS071_CONSTRAINTS)
    assert loaded.statistics.compiled == 2
    assert solved(
        loaded,
        known.start,
        variable_bounds=known.variable_bounds,
        constraint_bounds=known.constraint_bounds,
    ).objective == pytest.approx(
        solved(
            compiled,
            known.start,
            variable_bounds=known.variable_bounds,
            constraint_bounds=known.constraint_bounds,
        ).objective,
        abs=1e-9,
    )


def test_linear_problems_and_feasibility_problems_survive() -> None:
    """Checks problems whose Hessian, objective or constraints are absent."""
    linear = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, p: x[0] + 2.0 * x[1],
        constraints=(lambda x, p: x[0] + x[1],),
        sense=xip.Sense.MAXIMIZE,
    )
    feasibility = xip.Problem(
        variables=xip.Variable(),
        evaluate=xip.evaluation_from_functions(objective=xip.no_objective),
    )
    cases = (
        (
            linear,
            (np.zeros(()),),
            xip.Bounds(0.0, 5.0),
            xip.Bounds(lower=(1.0,)),
        ),
        (feasibility, None, xip.Bounds(0.0, 1.0), None),
    )
    for problem, constraints, variable_bounds, constraint_bounds in cases:
        compiled = xip.compile(
            problem, options=xip.CompileOptions(hessian=xip.Hessian.LIMITED_MEMORY)
        )
        loaded = xip.loads(
            xip.dumps(compiled), problem.variables, constraints=constraints
        )
        assert loaded.sense is problem.sense
        assert loaded.options.hessian is xip.Hessian.LIMITED_MEMORY
        solution = solved(
            loaded,
            variable_bounds=variable_bounds,
            constraint_bounds=constraint_bounds,
        )
        assert solution.status.succeeded


def test_dict_constraints_come_back_as_a_dict(tmp_path: Path) -> None:
    """Checks that a file with dictionary constraints loads with a matching tree and returns a dict."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, p: jnp.sum(x**2),
        constraints={
            "sum": lambda x, p: jnp.sum(x),
            "product": lambda x, p: x[0] * x[1],
        },
    )
    constraint_bounds = xip.Bounds(
        lower={"sum": 1.0, "product": 0.0},
        upper={"sum": 3.0, "product": np.inf},
    )
    compiled = xip.compile(problem)
    xip.save(compiled, tmp_path / "dict.xip")
    with pytest.raises(ValueError):
        xip.load(
            tmp_path / "dict.xip",
            problem.variables,
            constraints=(np.zeros(()), np.zeros(())),
        )
    loaded = xip.load(
        tmp_path / "dict.xip",
        problem.variables,
        constraints={"sum": np.zeros(()), "product": np.zeros(())},
    )
    start = np.array([1.0, 1.0])
    first = solved(compiled, start, constraint_bounds=constraint_bounds)
    second = solved(loaded, start, constraint_bounds=constraint_bounds)
    assert isinstance(second.constraints, dict)
    assert set(second.constraints) == {"sum", "product"}
    assert second.objective == pytest.approx(first.objective, abs=1e-8)
    assert second.objective == pytest.approx(0.5, abs=1e-6)
    assert second.constraints["sum"] == pytest.approx(1.0, abs=1e-6)


def test_pickling_is_refused() -> None:
    """Checks that a compiled problem points to `dumps` instead of pickling."""
    import pickle

    compiled = xip.compile(hs071().problem)
    with pytest.raises(TypeError, match="xipopt.dumps"):
        pickle.dumps(compiled)


def test_damaged_files_are_rejected(tmp_path: Path) -> None:
    """Checks the errors for a file that is not a problem, cut off or of another kind."""
    known = hs071()
    data = xip.dumps(xip.compile(known.problem))
    with pytest.raises(cyxla.BundleError, match="not a bundle"):
        xip.loads(b"x" * 100, known.problem.variables)
    with pytest.raises(cyxla.BundleError, match="cut off"):
        xip.loads(data[:-100], known.problem.variables)
    other = cyxla.BundleWriter("other.object", 1).to_bytes({})
    with pytest.raises(cyxla.BundleError, match="holds other.object"):
        xip.loads(other, known.problem.variables)


def test_start_point_is_inside_the_bounds() -> None:
    """Checks the point nearest to the origin for a finite bound, a half-line and a free variable."""
    layout = variable_layout(xip.Variable(shape=(4,)))
    lower, upper = _bounds.bounds_vectors(
        xip.Bounds(
            lower=np.array([0.0, 1.0, -np.inf, -np.inf]),
            upper=np.array([2.0, np.inf, 3.0, np.inf]),
        ),
        layout,
        "variable_bounds",
        "variables",
    )
    np.testing.assert_array_equal(
        _bounds.start_point(lower, upper), [0.0, 1.0, 0.0, 0.0]
    )


def test_warm_up_parameters_use_values_or_zeros() -> None:
    """Checks that given parameters are used, and that shapes alone give zeros."""
    params = {"a": np.array([1.0, 2.0])}
    layout = params_layout(params)
    np.testing.assert_array_equal(
        _serialize.warm_up_parameters(layout, params), [1.0, 2.0]
    )
    shapes = {"a": jax.ShapeDtypeStruct((2,), jnp.float64)}
    np.testing.assert_array_equal(
        _serialize.warm_up_parameters(layout, shapes), [0.0, 0.0]
    )


class FailingExecutable(NamedTuple):
    """A stand-in for an executable that fails when it runs, and notes what it ran on."""

    input_types: tuple[cyxla.ArrayType, ...]
    runs: list[list[np.ndarray]]

    def warm_up(self, arguments: Any) -> None:
        """Records the arguments and fails like a program that rejects its input."""
        self.runs.append(list(arguments))
        raise RuntimeError("the matrix is singular")


def test_failed_warm_up_is_a_warning() -> None:
    """Checks that a program that fails on the warm-up input does not stop loading."""
    point = np.full(4, 3.0)
    float_type = np.dtype("float64")
    types = (
        cyxla.ArrayType(float_type, (4,)),
        cyxla.ArrayType(float_type, (1,)),
        cyxla.ArrayType(float_type, ()),
        cyxla.ArrayType(float_type, (2,)),
    )
    executable = FailingExecutable(types, [])
    with pytest.warns(RuntimeWarning, match="hessian program failed.*singular"):
        _serialize.warm_up_program(
            Program.HESSIAN,
            cast(cyxla.Executable, executable),
            point,
            params_layout(None),
            None,
        )
    (arguments,) = executable.runs
    np.testing.assert_array_equal(arguments[0], [3.0, 3.0, 3.0, 3.0])
    assert [argument.shape for argument in arguments] == [(4,), (1,), (), (2,)]


def test_warm_up_passes_silently_for_a_normal_problem() -> None:
    """Checks that loading with the warm-up raises no warning when the programs run."""
    known = hs071()
    data = xip.dumps(xip.compile(known.problem))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xip.loads(
            data,
            known.problem.variables,
            constraints=HS071_CONSTRAINTS,
            variable_bounds=known.variable_bounds,
        )


@pytest.mark.parametrize("custom_calls", ["detect", ("first", "second"), ()])
def test_compile_options_survive_json(
    custom_calls: tuple[str, ...] | Literal["detect"],
) -> None:
    """Checks that the options of a problem are restored from the header of a file."""
    options = xip.CompileOptions(custom_calls=custom_calls, debug_info=False)
    data = json.loads(json.dumps(cyxla.encode_record(options)))
    assert cyxla.decode_record(xip.CompileOptions, data) == options
    assert cyxla.decode_record(xip.CompileOptions, {"zero_copy": False}) == (
        xip.CompileOptions(zero_copy=False)
    )


def test_problem_without_debug_info_is_smaller() -> None:
    """Checks that removing the source locations shrinks the file and keeps the solution."""
    known = hs071()
    sizes = {}
    solutions = {}
    for debug_info in (True, False):
        options = xip.CompileOptions(debug_info=debug_info, disk_cache=False)
        compiled = xip.compile(known.problem, options=options)
        sizes[debug_info] = len(xip.dumps(compiled))
        solutions[debug_info] = solved(
            compiled,
            known.start,
            variable_bounds=known.variable_bounds,
            constraint_bounds=known.constraint_bounds,
        )
    assert sizes[False] < 0.9 * sizes[True]
    np.testing.assert_allclose(solutions[False].x, solutions[True].x, atol=1e-10)


@pytest.mark.parametrize("workers", [1, 2, None])
def test_programs_load_on_any_number_of_threads(workers: int | None) -> None:
    """Checks that the number of threads changes neither the programs nor the solution."""
    known = hs071()
    data = xip.dumps(xip.compile(known.problem))
    loaded = xip.loads(
        data,
        known.problem.variables,
        constraints=HS071_CONSTRAINTS,
        variable_bounds=known.variable_bounds,
        workers=workers,
    )
    assert loaded.statistics.loaded == 2
    solution = solved(
        loaded,
        known.start,
        variable_bounds=known.variable_bounds,
        constraint_bounds=known.constraint_bounds,
    )
    assert solution.objective == pytest.approx(known.objective, abs=1e-6)


def test_load_programs_keeps_the_gaps_and_raises_failures() -> None:
    """Checks that absent programs stay absent and that a failing thread raises its error."""
    known = hs071()
    compiled = xip.compile(
        known.problem, options=xip.CompileOptions(hessian=xip.Hessian.LIMITED_MEMORY)
    )
    data = xip.dumps(compiled)
    bundle = cyxla.Bundle(data, _serialize.KIND, _serialize.VERSION)
    records = bundle.meta["programs"]
    assert records[-1] is None
    loaded = _serialize.load_programs(bundle, records, "raise", 4)
    assert [item is None for item in loaded] == [False, True]
    elsewhere = cyxla.TargetFingerprint("cpu", "other", "arm64", "plugin", 1, "0.1")
    with cyxla.override_target(elsewhere):
        with pytest.raises(
            cyxla.TargetMismatchError, match="no executable for this machine"
        ):
            _serialize.load_programs(bundle, records, "raise", 4)


def test_file_with_constraints_needs_the_constraints_tree() -> None:
    """Checks the error for loading a problem with constraints without a tree for them."""
    data = xip.dumps(xip.compile(hs071().problem))
    with pytest.raises(
        ValueError, match="has constraints, so `constraints` is required"
    ):
        xip.loads(data, hs071().problem.variables)


def test_bounds_of_an_empty_layout_are_empty() -> None:
    """Checks that bounds over a layout without entries give empty vectors."""
    empty = tree_layout(())
    lower, upper = _bounds.bounds_vectors(
        xip.Bounds(0.0, 1.0), empty, "bounds", "items"
    )
    assert lower.shape == upper.shape == (0,)


def test_leaf_shapes_of_numbers_and_lists() -> None:
    """Checks that numbers and lists in a tree given to `load` have shapes like arrays."""
    assert leaf_shape(3.0) == ()
    assert leaf_shape([1.0, 2.0]) == (2,)
