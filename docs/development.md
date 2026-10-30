# Development

## Setup and tests

```bash
git clone https://github.com/nardi/xipopt
cd xipopt
uv sync
uv run pytest
```

The tests are in two groups. The unit tests in `tests/unit` solve small problems and cover
all the code, and `uv run pytest -m unit` runs only them. The integration tests in
`tests/integration` compare with SciPy, solve published problems and start new processes,
and the code blocks of the README and of these pages count as integration tests as well.
Run them with `uv run pytest -m integration`.

`uv sync` builds the extension module in editable mode. Meson downloads the `meson-jll`
tool and the JLL libraries during the first configuration. The unit tests need 100%
coverage, including the Cython module, whose lines the development build traces. The code
blocks of the README and of these pages carry a `<!-- name: test_... -->` comment.

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
```

The traced build makes the callbacks several times slower. For timing, reinstall without
tracing. The `--no-config` flag stops uv from adding the coverage setting of
`pyproject.toml`, and `uv run --no-sync` stops it from rebuilding with that setting:

```bash
uv pip install --no-config --no-deps --no-build-isolation --reinstall -e . \
    -C setup-args=-Dcoverage=false
```

## Layout

- `src/xipopt/_solver.py` is the Cython module in pure Python mode. `_solver.pxd` holds
  the state that the callbacks share, `_ipopt.pxd` declares the C interface of Ipopt and
  `_solver.pyi` types the part that Python sees.
- `_platform.h` holds the operating system calls, so the module stays portable.
- `_compile.py` builds a problem, `_programs.py` holds the functions and derivatives it
  compiles, `_patterns.py` converts sparsity patterns and `_cache.py` keys the disk
  cache. `_solve.py` runs a solve and `_layout.py` flattens pytrees.
- `build_backend/xipopt_build.py` wraps meson-python. Meson reads the wraps of
  `subprojects` before `meson-jll sync` has written them, so the backend configures the
  project once in a scratch directory to write them first.
- `licenses/` holds the licenses of the libraries that the wheels bundle.

## Conventions

- Code is formatted and linted with Ruff and type-checked with `ty`. Avoid inheritance,
  prefer protocols and plain functions, and use `match` statements where they replace
  chains of conditions.
- The Cython code uses the pure Python syntax. The callbacks of Ipopt must not raise or
  allocate Python objects.
- Bump the pin on cyxla in `pyproject.toml` whenever its `ABI_VERSION` changes.

## Continuous integration

One workflow tests on Linux, macOS and Windows with the oldest and the newest supported
Python, lints, type checks, builds wheels with cibuildwheel for every platform and
builds the source distribution. A tag `vX.Y.Z` that matches the version in
`pyproject.toml` publishes to PyPI through trusted publishing, and so does a manual run
with the release input. A second workflow builds this site and deploys it to GitHub
Pages.
