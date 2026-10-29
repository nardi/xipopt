"""Tests of the shapes of problems: pytrees, parameters, senses and bounds."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xipopt as xip


def test_pytree_variables_and_parameters() -> None:
    """Checks that variables and parameters keep their structure and the types of parameters."""
    problem = xip.Problem(
        variables={
            "position": xip.Variable(lower=-3.0, upper=3.0, shape=(2,)),
            "scale": (xip.Variable(lower=0.5, upper=2.0),),
        },
        objective=lambda x, params: (
            jnp.sum((x["position"] - params["target"]) ** 2)
            + (x["scale"][0] - params["count"]) ** 2
        ),
        constraints=(),
    )
    params = {"target": np.array([1.0, -1.0]), "count": np.int32(1)}
    solution = xip.solve(problem, params=params)
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
        variables=xip.Variable(lower=-2.0, upper=2.0),
        objective=lambda x, params: -((x - 0.5) ** 2) + 3.0,
        sense=xip.Sense.MAXIMIZE,
    )
    solution = xip.solve(problem)
    assert solution.objective == pytest.approx(3.0)
    assert solution.x == pytest.approx(0.5)


def test_feasibility_problem_has_no_objective() -> None:
    """Checks that a problem without an objective returns a feasible point."""
    problem = xip.Problem(
        variables=xip.Variable(lower=0.0, upper=10.0, shape=(2,)),
        constraints=(
            xip.Constraint(lambda x, params: x[0] + x[1], lower=4.0, upper=4.0),
            xip.Constraint(lambda x, params: x[0] - x[1], lower=1.0),
        ),
    )
    solution = xip.solve(problem, np.array([1.0, 1.0]))
    assert solution.status.succeeded
    assert solution.objective == 0.0
    assert solution.constraints[0] == pytest.approx(4.0)
    assert solution.constraints[1] >= 1.0 - 1e-6


def test_vector_constraints_keep_their_shape() -> None:
    """Checks constraints with matrix outputs and bounds that broadcast to them."""
    problem = xip.Problem(
        variables=xip.Variable(lower=-5.0, upper=5.0, shape=(6,)),
        objective=lambda x, params: jnp.sum((x - 1.0) ** 2),
        constraints=(
            xip.Constraint(
                lambda x, params: x.reshape(2, 3),
                lower=np.array([[-1.0], [-2.0]]),
                upper=np.array([0.5, 0.5, 0.5]),
            ),
        ),
    )
    solution = xip.solve(problem)
    assert solution.constraints[0].shape == (2, 3)
    assert solution.constraint_multipliers[0].shape == (2, 3)
    np.testing.assert_allclose(solution.x, 0.5, atol=1e-6)


def test_start_defaults_to_the_origin_inside_the_bounds() -> None:
    """Checks the default starting point for bounds that exclude the origin."""
    problem = xip.Problem(
        variables=[
            xip.Variable(lower=2.0, upper=3.0),
            xip.Variable(lower=-3.0, upper=-1.0),
        ]
    )
    compiled = xip.compile(problem)
    np.testing.assert_allclose(compiled.variables.start(), [2.0, -1.0])


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
        variables=xip.Variable(lower=0.0, shape=(2,)),
        objective=lambda x, params: x[0] + 2.0 * x[1],
        constraints=(xip.Constraint(lambda x, params: x[0] + x[1], lower=1.0),),
    )
    compiled = xip.compile(problem)
    assert compiled.structure.hessian_rows.size == 0
    solution = xip.solve(compiled)
    assert solution.objective == pytest.approx(1.0)
    assert solution.statistics.hessian.calls == 0


def test_constraint_that_ignores_the_variables() -> None:
    """Checks the error for constraints with a Jacobian that has no entries."""
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum(x**2),
        constraints=(xip.Constraint(lambda x, params: jnp.asarray(1.0), lower=0.0),),
    )
    with pytest.raises(ValueError, match="do not depend on the variables"):
        xip.compile(problem)


def test_dense_derivatives_without_asdex_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that the automatic choice falls back to dense derivatives with a warning."""
    monkeypatch.setattr("xipopt._programs.asdex_installed", lambda: False)
    problem = xip.Problem(
        variables=xip.Variable(shape=(2,)),
        objective=lambda x, params: jnp.sum((x - 1.0) ** 2),
        constraints=(xip.Constraint(lambda x, params: x[0] * x[1], lower=0.0),),
    )
    with pytest.warns(UserWarning, match="derivatives are dense"):
        compiled = xip.compile(problem, options=xip.CompileOptions(disk_cache=False))
    assert compiled.structure.jacobian_rows.size == 2
    assert compiled.structure.hessian_rows.size == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solution = xip.solve(compiled)
    np.testing.assert_allclose(solution.x, 1.0, atol=1e-6)


def linear_system_problem() -> xip.Problem:
    """Returns a problem whose objective solves a linear system, which asdex cannot trace."""
    return xip.Problem(
        variables=xip.Variable(lower=0.5, upper=2.0, shape=(3,)),
        objective=lambda x, params: jnp.sum(
            jnp.linalg.solve(jnp.diag(x) + jnp.eye(3), jnp.ones(3)) + x**2
        ),
    )


def test_detection_failure_names_the_ways_out() -> None:
    """Checks the error for a function whose sparsity asdex cannot detect."""
    pytest.importorskip("asdex")
    with pytest.raises(xip.SparsityDetectionError) as error:
        xip.compile(
            linear_system_problem(), options=xip.CompileOptions(disk_cache=False)
        )
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
    solution = xip.solve(linear_system_problem(), compile_options=options)
    assert solution.status.succeeded
    np.testing.assert_allclose(solution.x, 0.5, atol=1e-5)
