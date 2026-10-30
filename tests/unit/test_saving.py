"""Tests of saving a compiled problem and loading it in the same and in another process."""

import dataclasses
from pathlib import Path
from typing import Any, NamedTuple

import cyxla
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import bratu_control, hs071

import xipopt as xip


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
