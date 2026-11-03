"""Tests of saving a compiled problem and loading it in the same and in another process."""

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import cyxla
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import bratu_control, hs071

import xipopt as xip
from xipopt import _serialize
from xipopt._layout import params_layout, variable_layout
from xipopt._types import Program


def solved(compiled: xip.CompiledProblem, start: Any = None) -> xip.Solution:
    """Solves a compiled problem with a tight tolerance."""
    return xip.solve(compiled, start, options=xip.SolveOptions(tolerance=1e-10))


def test_loaded_problem_solves_like_the_original(tmp_path: Path) -> None:
    """Checks that a problem saved to a file gives the same solution after loading."""
    known = hs071()
    compiled = xip.compile(known.problem)
    xip.save(compiled, tmp_path / "hs071.xip")
    loaded = xip.load(tmp_path / "hs071.xip", known.problem.variables)
    assert loaded.problem is None
    assert loaded.sense is compiled.sense
    assert loaded.options == compiled.options
    assert loaded.constraint_shapes == compiled.constraint_shapes
    assert loaded.statistics.loaded == 5
    assert loaded.statistics.compiled == 0
    first, second = solved(compiled, known.start), solved(loaded, known.start)
    assert second.status is first.status
    assert second.objective == pytest.approx(first.objective, abs=1e-12)
    np.testing.assert_allclose(second.x, first.x, atol=1e-10)
    assert second.iterations == first.iterations


def test_bytes_round_trip_with_pytrees_and_parameters() -> None:
    """Checks a problem with dictionary variables and parameters, loaded from bytes."""
    base = bratu_control(10)
    objective = base.objective
    assert objective is not None
    params = {"scale": jax.ShapeDtypeStruct((), jnp.float64)}
    problem = base._replace(objective=lambda x, p: p["scale"] * objective(x, p))
    compiled = xip.compile(problem, params)
    data = xip.dumps(compiled)
    template = {
        "u": xip.Variable(shape=(10,)),
        "c": xip.Variable(shape=(10,)),
    }
    loaded = xip.loads(data, template, {"scale": np.float64(1.0)})
    assert [leaf.path for leaf in loaded.variables.fingerprint] == ["['c']", "['u']"]
    original = xip.solve(compiled, params={"scale": 2.0})
    again = xip.solve(loaded, params={"scale": 2.0})
    assert again.objective == pytest.approx(original.objective, rel=1e-9)
    assert set(again.x) == {"u", "c"}
    np.testing.assert_allclose(again.x["u"], original.x["u"], atol=1e-8)


def test_arrays_can_stand_in_for_variables() -> None:
    """Checks that a template of arrays works, and that the saved bounds are used."""
    problem = xip.Problem(
        variables={"a": xip.Variable(lower=1.0, upper=2.0, shape=(2,))},
        objective=lambda x, p: jnp.sum(x["a"]),
    )
    data = xip.dumps(xip.compile(problem))
    loaded = xip.loads(data, {"a": np.zeros(2)}, warm_up=False)
    np.testing.assert_allclose(xip.solve(loaded).x["a"], 1.0, atol=1e-6)


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
    variables = Plant(xip.Variable(lower=-3.0, upper=3.0), xip.Variable(shape=(2,)))
    problem = xip.Problem(
        variables=variables,
        objective=lambda x, p: (
            (x.flow - p.offset) ** 2 + p.weight * jnp.sum(x.count**2)
        ),
    )
    params = Settings(np.float64(1.0), np.float64(0.0))
    loaded = xip.loads(
        xip.dumps(xip.compile(problem, params)),
        Plant(np.zeros(()), np.zeros(2)),
        Settings(jax.ShapeDtypeStruct((), jnp.float64), np.float64(0.0)),
    )
    solution = xip.solve(loaded, params=Settings(1.0, 2.0))
    assert isinstance(solution.x, Plant)
    assert solution.x.flow == pytest.approx(2.0, abs=1e-6)


def test_trees_are_checked_against_the_file() -> None:
    """Checks the errors for variables and parameters that do not match."""
    problem = xip.Problem(
        variables={"a": xip.Variable(shape=(2,)), "b": xip.Variable()},
        objective=lambda x, p: jnp.sum(x["a"]) ** 2 + x["b"] ** 2 + jnp.sum(p["w"]),
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
            xip.loads(data, known.problem.variables, on_mismatch="raise")
        loaded = xip.loads(data, known.problem.variables)
    assert loaded.statistics.compiled == 5
    assert solved(loaded, known.start).objective == pytest.approx(
        solved(compiled, known.start).objective, abs=1e-9
    )


def test_linear_problems_and_feasibility_problems_survive() -> None:
    """Checks problems whose Hessian, objective or constraints are absent."""
    linear = xip.Problem(
        variables=xip.Variable(lower=0.0, upper=5.0, shape=(2,)),
        objective=lambda x, p: x[0] + 2.0 * x[1],
        constraints=(xip.Constraint(lambda x, p: x[0] + x[1], lower=1.0),),
        sense=xip.Sense.MAXIMIZE,
    )
    feasibility = xip.Problem(variables=xip.Variable(lower=0.0, upper=1.0))
    for problem in (linear, feasibility):
        compiled = xip.compile(
            problem, options=xip.CompileOptions(hessian=xip.Hessian.LIMITED_MEMORY)
        )
        loaded = xip.loads(xip.dumps(compiled), problem.variables)
        assert loaded.sense is problem.sense
        assert loaded.options.hessian is xip.Hessian.LIMITED_MEMORY
        assert solved(loaded).status.succeeded


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


def test_warm_up_point_is_inside_the_bounds() -> None:
    """Checks the midpoint, the single finite bound and the free variable."""
    layout = variable_layout(
        xip.Variable(
            lower=np.array([0.0, 1.0, -np.inf, -np.inf]),
            upper=np.array([2.0, np.inf, 3.0, np.inf]),
        )
    )
    np.testing.assert_array_equal(
        _serialize.warm_up_point(layout), [1.0, 1.0, 3.0, 0.0]
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
    known = hs071()
    layout = variable_layout(known.problem.variables)
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
            layout,
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
        xip.loads(data, known.problem.variables)


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
        solutions[debug_info] = solved(compiled, known.start)
    assert sizes[False] < 0.9 * sizes[True]
    np.testing.assert_allclose(solutions[False].x, solutions[True].x, atol=1e-10)


@pytest.mark.parametrize("workers", [1, 2, None])
def test_programs_load_on_any_number_of_threads(workers: int | None) -> None:
    """Checks that the number of threads changes neither the programs nor the solution."""
    known = hs071()
    data = xip.dumps(xip.compile(known.problem))
    loaded = xip.loads(data, known.problem.variables, workers=workers)
    assert loaded.statistics.loaded == 5
    solution = solved(loaded, known.start)
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
    assert [item is None for item in loaded] == [False] * 4 + [True]
    elsewhere = cyxla.TargetFingerprint("cpu", "other", "arm64", "plugin", 1, "0.1")
    with cyxla.override_target(elsewhere):
        with pytest.raises(
            cyxla.TargetMismatchError, match="no executable for this machine"
        ):
            _serialize.load_programs(bundle, records, "raise", 4)
