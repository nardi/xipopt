# Derivatives and sparsity

Ipopt needs the gradient of the objective, the Jacobian of the constraints and the
Hessian of the Lagrangian. The last two are matrices, and for most large problems they
are mostly zeros. Ipopt stores and factorizes only their nonzero entries, so a problem
solves fast when xipopt tells it where those are.

## What gets compiled

xipopt compiles five programs from the functions of a problem.

| Ipopt callback | Program |
| --- | --- |
| `eval_f` | The objective |
| `eval_grad_f` | The gradient of the objective |
| `eval_g` | All constraints, concatenated |
| `eval_jac_g` | The Jacobian, at its nonzero entries |
| `eval_h` | The Hessian of the Lagrangian, at the nonzero entries of its lower triangle |

The Lagrangian is `factor * objective + multipliers @ constraints`, with the factor and
the multipliers as inputs of the program. The entries of the Jacobian and the Hessian
come out in a fixed order that the program and the structure given to Ipopt share.

## Detection

With the default `derivatives=Derivatives.AUTOMATIC`, xipopt uses
[asdex](https://pypi.org/project/asdex/) when it is installed. asdex traces a function
and finds which outputs depend on which inputs, colors the columns of the Jacobian so
that columns without a shared row can be computed together, and evaluates a compressed
Jacobian with few forward or reverse passes. It does the same for the Hessian with
symmetric coloring and Hessian-vector products.

<!-- name: test_docs_derivatives -->
```python
import jax.numpy as jnp
import numpy as np

import xipopt as xip

SIZE = 8
spacing = 1.0 / (SIZE + 1)


def objective(x, params):
    return spacing * jnp.sum((x["u"] - 0.1) ** 2 + 0.01 * x["c"] ** 2)


def equation(x, params):
    # The second difference of u, a tridiagonal operator, minus a control and a source.
    padded = jnp.concatenate([jnp.zeros(1), x["u"], jnp.zeros(1)])
    laplacian = (2.0 * x["u"] - padded[:-2] - padded[2:]) / spacing**2
    return laplacian - jnp.exp(x["u"]) - x["c"]


problem = xip.Problem(
    variables={"u": xip.Variable(shape=(SIZE,)), "c": xip.Variable(shape=(SIZE,))},
    objective=objective,
    constraints=(xip.Constraint(equation, lower=0.0, upper=0.0),),
)
compiled = xip.compile(problem)
print(compiled.structure.jacobian_rows.size, compiled.structure.hessian_rows.size)
```

<!-- name: test_docs_derivatives -->
```python
assert compiled.structure.jacobian_rows.size == 4 * SIZE - 2
assert compiled.structure.hessian_rows.size == 2 * SIZE
```

The Jacobian of this problem has three entries per row for the state and one for the
control, and the Hessian is diagonal. A dense Jacobian would have `8 * 16` entries.

Dictionaries flatten in the order of their keys, so the control `c` comes before the
state `u` in the columns of the Jacobian. Patterns use the order of the flattened
variables, and the rows are the entries of the constraints in order.

## Given patterns

Detection traces the functions with abstract values, so it cannot see through
everything, and it costs time on a large problem. A problem can give the patterns itself
as `jacobian_pattern` and `hessian_pattern`, in any of four forms.

- A boolean matrix.
- A pair of integer vectors `(rows, columns)`.
- An `asdex.SparsityPattern`.
- A `jax.experimental.sparse.BCOO` matrix, whose stored entries count as nonzero.

A Hessian pattern is square, and it may list one triangle or both. xipopt uses the lower
triangle, which is the part Ipopt asks for.

<!-- name: test_docs_derivatives -->
```python
rows, columns = [], []
for row in range(SIZE):
    rows.append(row)
    columns.append(row)
    for neighbor in (row - 1, row, row + 1):
        if 0 <= neighbor < SIZE:
            rows.append(row)
            columns.append(SIZE + neighbor)

given = problem._replace(
    jacobian_pattern=(np.array(rows), np.array(columns)),
    hessian_pattern=(np.arange(2 * SIZE), np.arange(2 * SIZE)),
)
solution = xip.solve(given)
```

<!-- name: test_docs_derivatives -->
```python
assert solution.status.succeeded
assert np.max(np.abs(solution.constraints[0])) < 1e-6
```

A pattern has to include every entry that can be nonzero. An entry that is left out is
not computed, and the solve then runs on a wrong derivative without any error. Extra
entries only cost time.

## Failed detection

Some functions are beyond what asdex can trace. A linear solve is one. In that case
compiling raises a [`SparsityDetectionError`][xipopt.SparsityDetectionError] that names
the ways out, and xipopt does not pick one for you:

- Give the patterns of the problem, as above.
- Use `CompileOptions(derivatives=Derivatives.DENSE)`, which differentiates densely with
  JAX and keeps all entries, or only the entries of a given pattern.
- Use `CompileOptions(hessian=Hessian.LIMITED_MEMORY)`, which lets Ipopt approximate the
  Hessian from gradients. No Hessian program is compiled, and the solve takes more
  iterations.

The third choice changes the algorithm and its convergence, which is why it is not made
silently. A dense pattern is not an error. It only means that the problem is as costly
as a dense one.

## Without asdex

asdex is a dependency, so it is normally there. When someone has uninstalled it,
`Derivatives.AUTOMATIC` computes dense derivatives and compiling warns.
`Derivatives.SPARSE` raises an `ImportError` instead, and `Derivatives.DENSE` is the
explicit choice. Dense derivatives still use the entries of a given pattern, so a
problem that gives its patterns gets sparse structure for Ipopt. Only the evaluation is
dense, which suits problems of a few hundred variables.
