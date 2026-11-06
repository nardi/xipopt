"""Tests of the shapes of problems: pytrees, parameters, senses and bounds."""

import warnings
from typing import Any

import cyxla
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import SolveBounds

import xipopt as xip


def test_pytree_variables_and_parameters() -> None:
    """Checks that variables and parameters keep their structure and the types of parameters."""
    problem = xip.Problem(
        variables={
            "position": xip.Variable(shape=(2,)),
            "scale": (xip.Variable(),),
        },
        objective=lambda x, params: (
            jnp.sum((x["position"] - params["target"]) ** 2)
            + (x["scale"][0] - params["count"]) ** 2
        ),
    )
    params = {"target": np.array([1.0, -1.0]), "count": np.int32(1)}
    solution = xip.solve(
        problem,
        params=params,
        variable_bounds=xip.Bounds(
            lower={"position": -3.0, "scale": (0.5,)},
            upper={"position": 3.0, "scale": (2.0,)},
        ),
    )
    assert solution.status.succeeded
    np.testing.assert_allclose(solution.x["position"], [1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(solution.x["scale"][0], 1.0, atol=1e-6)
    assert solution.constraints == ()
    assert solution.objective == pytest.approx(0.0, abs=1e-9)


def test_compile_accepts_shapes_as_parameters() -> None:
    """Checks that parameters can be given as `jax.ShapeDtypeStruct`s."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum((x - params) ** 2),
    )
    compiled = xip.compile(problem, jax.ShapeDtypeStruct((2,), jnp.float32))
    solution = xip.solve(compiled, params=np.array([2.0, 3.0]))
    np.testing.assert_allclose(solution.x, [2.0, 3.0], atol=1e-6)


def test_maximize_reports_the_objective_of_the_problem() -> None:
    """Checks that a maximization returns the maximum and not its negation."""
    problem = xip.Problem(
        variables=xip.Variable(),
        objective=lambda x, params: -((x - 0.5) ** 2) + 3.0,
        sense=xip.Sense.MAXIMIZE,
    )
    solution = xip.solve(problem, variable_bounds=xip.Bounds(-2.0, 2.0))
    assert solution.objective == pytest.approx(3.0)
    assert solution.x == pytest.approx(0.5)


def test_feasibility_problem_has_no_objective() -> None:
    """Checks that a problem without an objective returns a feasible point."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        constraints=(
            lambda x, params: x[0] + x[1],
            lambda x, params: x[0] - x[1],
        ),
    )
    solution = xip.solve(
        problem,
        np.array([1.0, 1.0]),
        variable_bounds=xip.Bounds(0.0, 10.0),
        constraint_bounds=xip.Bounds(lower=(4.0, 1.0), upper=(4.0, np.inf)),
    )
    assert solution.status.succeeded
    assert solution.objective == 0.0
    assert solution.constraints[0] == pytest.approx(4.0)
    assert solution.constraints[1] >= 1.0 - 1e-6


def test_vector_constraints_keep_their_shape() -> None:
    """Checks constraints with matrix outputs and bounds that broadcast to them."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(6,)),
        objective=lambda x, params: jnp.sum((x - 1.0) ** 2),
        constraints=(lambda x, params: x.reshape(2, 3),),
    )
    solution = xip.solve(
        problem,
        variable_bounds=xip.Bounds(-5.0, 5.0),
        constraint_bounds=xip.Bounds(
            lower=(np.array([[-1.0], [-2.0]]),),
            upper=(np.array([0.5, 0.5, 0.5]),),
        ),
    )
    assert solution.constraints[0].shape == (2, 3)
    assert solution.constraint_multipliers[0].shape == (2, 3)
    np.testing.assert_allclose(solution.x, 0.5, atol=1e-6)


def test_start_defaults_to_the_origin_inside_the_bounds() -> None:
    """Checks that a solve without `x0` starts from the point of the bounds nearest the origin."""
    starts: list[list[float]] = []

    def record(point: list[np.ndarray]) -> None:
        starts.append([float(value) for value in point])

    def objective(x, params):
        jax.debug.callback(record, x)
        return x[0] ** 2 + x[1] ** 2

    problem = xip.Problem(
        variables=[xip.Variable(), xip.Variable()],
        objective=objective,
    )
    options = xip.CompileOptions(
        host_callbacks=True, derivatives=xip.Derivatives.DENSE, disk_cache=False
    )
    solution = xip.solve(
        problem,
        variable_bounds=xip.Bounds([2.0, -3.0], [3.0, -1.0]),
        compile_options=options,
    )
    assert starts[0] == [2.0, -1.0]
    np.testing.assert_allclose(solution.x, [2.0, -1.0], atol=1e-6)


def test_infinite_bounds_are_free() -> None:
    """Checks that variables without bounds can take large values."""
    problem = xip.Problem(
        variables=xip.Variable(),
        objective=lambda x, params: (x - 1e6) ** 2 / 1e12,
    )
    assert xip.solve(problem).x == pytest.approx(1e6, rel=1e-6)


def test_linear_problem_without_hessian_entries() -> None:
    """Checks a linear program, whose Hessian has no entries and no program."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: x[0] + 2.0 * x[1],
        constraints=(lambda x, params: x[0] + x[1],),
    )
    compiled = xip.compile(problem)
    assert compiled.structure.hessian_rows.size == 0
    solution = xip.solve(
        compiled,
        variable_bounds=xip.Bounds(lower=0.0),
        constraint_bounds=xip.Bounds(lower=1.0),
    )
    assert solution.objective == pytest.approx(1.0)
    assert solution.statistics.hessian_program.executions == 0


def test_constraint_that_ignores_the_variables() -> None:
    """Checks the error for constraints with a Jacobian that has no entries."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum(x**2),
        constraints=(lambda x, params: jnp.asarray(1.0),),
    )
    with pytest.raises(ValueError, match="do not depend on the variables"):
        xip.compile(problem)


def test_dense_derivatives_without_asdex_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that the automatic choice falls back to dense derivatives with a warning."""
    monkeypatch.setattr("xipopt._programs.asdex_installed", lambda: False)
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum((x - 1.0) ** 2),
        constraints=(lambda x, params: x[0] * x[1],),
    )
    with pytest.warns(UserWarning, match="derivatives are dense"):
        compiled = xip.compile(problem, options=xip.CompileOptions(disk_cache=False))
    assert compiled.structure.jacobian_rows.size == 2
    assert compiled.structure.hessian_rows.size == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solution = xip.solve(compiled, constraint_bounds=xip.Bounds(lower=0.0))
    np.testing.assert_allclose(solution.x, 1.0, atol=1e-6)


def root_problem() -> xip.Problem:
    """Returns a problem whose objective solves a linear system, which asdex cannot trace."""

    def inverse(x):
        return jax.lax.custom_linear_solve(
            lambda v: (x + 1.0) * v,
            jnp.ones_like(x),
            solve=lambda matvec, b: b / (x + 1.0),
            symmetric=True,
        )

    return xip.Problem(
        variables=xip.Variable(shape=(3,)),
        objective=lambda x, params: jnp.sum(inverse(x) + x**2),
    )


ROOT_VARIABLE_BOUNDS = xip.Bounds(0.5, 2.0)
"""The bounds of the variables of `root_problem`."""


def test_detection_failure_names_the_ways_out() -> None:
    """Checks the error for a function whose sparsity asdex cannot detect."""
    with pytest.raises(xip.SparsityDetectionError) as error:
        xip.compile(root_problem(), options=xip.CompileOptions(disk_cache=False))
    assert "Derivatives.DENSE" in str(error.value)
    assert "LIMITED_MEMORY" in str(error.value)


@pytest.mark.parametrize(
    "options",
    [
        xip.CompileOptions(derivatives=xip.Derivatives.DENSE),
        xip.CompileOptions(hessian=xip.Hessian.LIMITED_MEMORY),
    ],
)
def test_ways_out_of_a_detection_failure_work(options: xip.CompileOptions) -> None:
    """Checks that dense derivatives and a limited-memory Hessian solve the same problem."""
    solution = xip.solve(
        root_problem(),
        compile_options=options,
        variable_bounds=ROOT_VARIABLE_BOUNDS,
    )
    assert solution.status.succeeded
    np.testing.assert_allclose(solution.x, 0.5, atol=1e-5)


def lowering_fails_with(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> pytest.ExceptionInfo[ValueError]:
    """Compiles a small problem while `cyxla.lower` raises a `ValueError` with the message."""

    def fail(*arguments: object, **options: object) -> None:
        raise ValueError(message)

    monkeypatch.setattr(cyxla, "lower", fail)
    problem = xip.Problem(
        xip.Variable(shape=(1,)), objective=lambda x, params: x[0] ** 2
    )
    with pytest.raises(ValueError) as caught:
        xip.compile(problem, options=xip.CompileOptions(disk_cache=False))
    return caught


def test_other_errors_of_lowering_are_not_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that only the error about custom calls gets the explanation of the option."""
    caught = lowering_fails_with(monkeypatch, "something else")
    assert str(caught.value) == "something else"


def test_error_about_custom_calls_names_the_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that the error of `cyxla.lower` is replaced by one that names `custom_calls`."""
    caught = lowering_fails_with(
        monkeypatch,
        "Cannot serialize code with custom calls whose targets have no "
        "compatibility guarantees: my_target",
    )
    assert "CompileOptions(custom_calls=" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        (xip.Bounds(-1.0, 1.0), [1.0, 1.0]),
        (xip.Bounds(np.array([0.0, -1.0]), np.array([2.0, 1.0])), [2.0, 1.0]),
        (xip.Bounds.from_array(np.array([[0.0, 2.0], [-1.0, 1.0]])), [2.0, 1.0]),
    ],
)
def test_bounds_as_a_number_or_arrays(
    bounds: xip.Bounds, expected: list[float]
) -> None:
    """Checks that a number, two arrays and an `(m, 2)` array give the same bounds."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum((x - 3.0) ** 2),
    )
    solution = xip.solve(problem, variable_bounds=bounds)
    np.testing.assert_allclose(solution.x, expected, atol=1e-6)


def test_bounds_as_pytrees_of_the_constraints() -> None:
    """Checks dict and tuple constraint bounds whose structure matches the constraints."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum((x - 3.0) ** 2),
        constraints={
            "sum": lambda x, params: x[0] + x[1],
            "difference": (lambda x, params: x[0] - x[1],),
        },
    )
    solution = xip.solve(
        problem,
        variable_bounds=xip.Bounds(0.0, 10.0),
        constraint_bounds=xip.Bounds(
            lower={"sum": -np.inf, "difference": (0.0,)},
            upper={"sum": 4.0, "difference": (0.0,)},
        ),
    )
    assert solution.status.succeeded
    np.testing.assert_allclose(solution.x, [2.0, 2.0], atol=1e-6)


def test_bounds_as_a_pytree_of_bounds() -> None:
    """Checks that a `Bounds` at every leaf gives the same answer as a `Bounds` of pytrees."""
    problem = xip.Problem(
        variables={"a": xip.Variable(), "b": xip.Variable(shape=(2,))},
        objective=lambda x, params: (x["a"] - 3.0) ** 2 + jnp.sum((x["b"] - 3.0) ** 2),
        constraints={
            "sum": lambda x, params: x["a"] + x["b"][0],
            "difference": (lambda x, params: x["b"][0] - x["b"][1],),
        },
    )
    compiled = xip.compile(problem)
    transposed = xip.solve(
        compiled,
        variable_bounds={
            "a": xip.Bounds(0.0, 10.0),
            "b": xip.Bounds(np.array([0.0, 1.0]), np.array([10.0, 10.0])),
        },
        constraint_bounds={
            "sum": xip.Bounds(upper=4.0),
            "difference": (xip.Bounds(0.0, 0.0),),
        },
    )
    regular = xip.solve(
        compiled,
        variable_bounds=xip.Bounds(
            lower={"a": 0.0, "b": np.array([0.0, 1.0])},
            upper={"a": 10.0, "b": 10.0},
        ),
        constraint_bounds=xip.Bounds(
            lower={"sum": -np.inf, "difference": (0.0,)},
            upper={"sum": 4.0, "difference": (0.0,)},
        ),
    )
    assert transposed.status.succeeded
    np.testing.assert_allclose(transposed.x["b"], regular.x["b"], atol=1e-6)
    np.testing.assert_allclose(transposed.x["a"], regular.x["a"], atol=1e-6)


def test_bounds_pytree_with_some_leaves_that_are_not_bounds_is_an_error() -> None:
    """Checks that a pytree mixing `Bounds` with other leaves raises a `ValueError`."""
    problem = xip.Problem(
        variables={"a": xip.Variable(), "b": xip.Variable()},
        objective=lambda x, params: x["a"] ** 2 + x["b"] ** 2,
    )
    with pytest.raises(ValueError, match="`variable_bounds` must be a `Bounds`"):
        xip.solve(problem, variable_bounds={"a": xip.Bounds(0.0, 1.0), "b": 1.0})


def test_one_compiled_problem_solves_for_two_sets_of_bounds() -> None:
    """Checks that the bounds of a solve change the answer for the same compiled problem."""
    compiled = xip.compile(
        xip.Problem(
            variables=xip.Variable(),
            objective=lambda x, params: (x - 3.0) ** 2,
        )
    )
    near = xip.solve(compiled, variable_bounds=xip.Bounds(0.0, 1.0))
    far = xip.solve(compiled, variable_bounds=xip.Bounds(4.0, 5.0))
    assert near.x == pytest.approx(1.0, abs=1e-6)
    assert far.x == pytest.approx(4.0, abs=1e-6)


def test_problem_takes_evaluate_or_functions_but_not_both() -> None:
    """Checks that `Problem` rejects mixed and missing ways of giving its functions."""
    variables = xip.Variable(shape=(2,))
    evaluate = xip.evaluation_from_functions(objective=lambda x, params: jnp.sum(x**2))
    # The call is typed `Any` because the type checker rejects the mixed arguments, which
    # is what this test provokes.
    problem_class: Any = xip.Problem

    def objective(x, params):
        return jnp.sum(x**2)

    def constraint(x, params):
        return x[0]

    with pytest.raises(TypeError, match="but not both"):
        problem_class(variables, evaluate, objective=objective)
    with pytest.raises(TypeError, match="but not both"):
        problem_class(variables, evaluate, constraints=(constraint,))
    with pytest.raises(TypeError, match="Give `evaluate`"):
        problem_class(variables)


def test_evaluate_and_separate_functions_give_the_same_solution() -> None:
    """Checks that `evaluate` and separate functions solve to the same point."""
    variables = xip.Variable(shape=(2,))

    def objective(x, params):
        return jnp.sum((x - 3.0) ** 2)

    def constraint(x, params):
        return x[0] + x[1]

    def evaluate(x, params):
        return xip.Evaluation(objective(x, params), (constraint(x, params),))

    bounds = SolveBounds(
        variable_bounds=xip.Bounds(0.0, 10.0),
        constraint_bounds=xip.Bounds(upper=(4.0,)),
    )
    from_evaluate = xip.solve(xip.Problem(variables, evaluate), **bounds)
    from_functions = xip.solve(
        xip.Problem(variables, objective=objective, constraints=(constraint,)),
        **bounds,
    )
    np.testing.assert_allclose(from_evaluate.x, [2.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(from_functions.x, [2.0, 2.0], atol=1e-6)


def test_evaluation_from_functions_takes_keyword_functions_only() -> None:
    """Checks that `evaluation_from_functions` rejects positional functions."""
    # Typed `Any` because the type checker rejects the positional call that is provoked.
    helper: Any = xip.evaluation_from_functions
    with pytest.raises(TypeError, match="positional"):
        helper(lambda x, params: jnp.sum(x**2))


def test_solution_constraints_keep_the_dict_structure() -> None:
    """Checks that the constraints and their multipliers are dictionaries like the constraints."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum((x - 3.0) ** 2),
        constraints={
            "sum": lambda x, params: x[0] + x[1],
            "difference": lambda x, params: x[0] - x[1],
        },
    )
    solution = xip.solve(
        problem,
        constraint_bounds=xip.Bounds(
            lower={"sum": -np.inf, "difference": 0.0},
            upper={"sum": 4.0, "difference": 0.0},
        ),
    )
    assert isinstance(solution.constraints, dict)
    assert set(solution.constraints) == {"sum", "difference"}
    assert isinstance(solution.constraint_multipliers, dict)
    assert set(solution.constraint_multipliers) == {"sum", "difference"}
    assert solution.constraints["sum"] == pytest.approx(4.0, abs=1e-6)


def bounds_problem() -> xip.Problem:
    """Returns a problem with two variables and one constraint, which needs bounds."""
    return xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum(x**2),
        constraints=(lambda x, params: x[0] + x[1],),
    )


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        (
            {
                "variable_bounds": xip.Bounds(1.0, 0.0),
                "constraint_bounds": xip.Bounds(0.0),
            },
            "`variable_bounds` has a lower bound above an upper bound.",
        ),
        (
            {
                "variable_bounds": xip.Bounds(np.zeros(3)),
                "constraint_bounds": xip.Bounds(0.0),
            },
            "does not broadcast",
        ),
        (
            {
                "variable_bounds": xip.Bounds((0.0, 0.0), (1.0, 1.0)),
                "constraint_bounds": xip.Bounds(0.0),
            },
            "`variable_bounds.lower` has structure",
        ),
        (
            {"variable_bounds": None},
            "`constraint_bounds` is required",
        ),
    ],
)
def test_bounds_errors_name_the_argument(keywords: dict, message: str) -> None:
    """Checks that a wrong `solve` bound raises a `ValueError` that names the argument."""
    with pytest.raises(ValueError, match=message):
        xip.solve(bounds_problem(), **keywords)


def test_constraint_bounds_are_not_allowed_without_constraints() -> None:
    """Checks that `constraint_bounds` raises an error for a problem without constraints."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum(x**2),
    )
    with pytest.raises(ValueError, match="not allowed for one without"):
        xip.solve(problem, constraint_bounds=xip.Bounds(0.0))
