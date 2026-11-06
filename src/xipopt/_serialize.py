"""Saving a compiled problem to a file and loading it in another process.

The file contains the compiled programs, the sparsity structure and the options. It does
not contain the functions of the problem or the trees of its variables and parameters. The
caller gives the trees when loading, which makes a loaded problem work exactly like a
compiled one, and the file only keeps the path and shape of each leaf to check them against.
"""

import importlib.metadata
import os
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import cyxla
import jax
import numpy as np
from jaxtyping import Float, PyTree

from xipopt import _cache
from xipopt._bounds import bounds_vectors, start_point
from xipopt._compile import CompiledProblem, check_output_sizes, make_native
from xipopt._layout import (
    ParamsLayout,
    layout_for,
    params_layout,
)
from xipopt._options import CompileOptions
from xipopt._problem_types import BoundsTree, is_variable
from xipopt._types import Program, Sense

KIND = "xipopt.problem"
"""The kind of bundle that contains a compiled problem."""

VERSION = 2
"""The version of the payload, which changes when the data in the file changes.

The payload holds the first-order program, the Hessian program, the sparsity structure and
the paths and shapes of the trees. It holds no bounds.
"""


def dumps(compiled: CompiledProblem) -> bytes:
    """Returns a compiled problem as bytes that `loads` reads.

    The programs are stored together with their StableHLO, so a machine that cannot use the
    stored executables compiles the source instead.
    """
    writer, header = pack(compiled)
    return writer.to_bytes(header)


def save(compiled: CompiledProblem, path: str | os.PathLike[str]) -> None:
    """Saves a compiled problem to a file that `load` reads. `.xip` is the usual extension."""
    writer, header = pack(compiled)
    writer.write(path, header)


def loads(
    data: bytes,
    variables: PyTree[Any],
    params: PyTree[Any] = None,
    *,
    constraints: PyTree[Any] = None,
    variable_bounds: BoundsTree | None = None,
    on_mismatch: Literal["recompile", "raise"] = "recompile",
    warm_up: bool = True,
    import_providers: bool = False,
    workers: int | None = None,
) -> CompiledProblem:
    """Returns the compiled problem that `dumps` made from bytes.

    `variables` and `params` are trees with the structure that the problem was compiled for.
    The leaves of `variables` are `Variable`s or arrays, and those of `params` are arrays or
    `jax.ShapeDtypeStruct`s, as for `compile`. A problem with constraints also needs
    `constraints`, a tree with the structure and shapes of `Evaluation.constraints`. Only the
    paths and shapes of the trees are checked against the file. The file holds no bounds, so
    they go to `solve`. The result solves like a compiled problem, with the structure of
    `variables` for `x`, except that its `problem` is `None`.

    Executables that were stored for another machine are replaced by a compilation of the
    StableHLO, unless `on_mismatch` is `raise`. With `warm_up` each program runs once, so
    the first solve is as fast as the later ones. The warm-up runs on the point nearest to
    the origin inside `variable_bounds` and on the given `params`, or on zeros for
    parameters that are only shapes. Without `variable_bounds` the point is zero. A program
    that fails there raises a warning and the problem loads anyway.

    A program that calls a library such as `klujax` needs that library imported before
    loading, since its handlers live in the library. Loading raises a `MissingTargetError`
    that names the library and its version when it is not. With `import_providers` loading
    imports those libraries itself, which runs their code, so only use it for files you
    trust.

    The programs of the file load on `workers` threads, which by default is one per program
    up to the number of processors. Pass 1 to load them one after another.
    """
    return read(
        cyxla.Bundle(data, KIND, VERSION),
        variables,
        params,
        constraints,
        variable_bounds,
        on_mismatch,
        warm_up,
        import_providers,
        workers,
    )


def load(
    path: str | os.PathLike[str],
    variables: PyTree[Any],
    params: PyTree[Any] = None,
    *,
    constraints: PyTree[Any] = None,
    variable_bounds: BoundsTree | None = None,
    on_mismatch: Literal["recompile", "raise"] = "recompile",
    warm_up: bool = True,
    import_providers: bool = False,
    workers: int | None = None,
) -> CompiledProblem:
    """Returns the compiled problem that `save` wrote to a file, as `loads` does."""
    return read(
        cyxla.Bundle.read(path, KIND, VERSION),
        variables,
        params,
        constraints,
        variable_bounds,
        on_mismatch,
        warm_up,
        import_providers,
        workers,
    )


def pack(compiled: CompiledProblem) -> tuple[cyxla.BundleWriter, dict[str, Any]]:
    """Returns the bundle writer and the header for a compiled problem.

    The programs are added to the writer, and their records go into the header.
    """
    if compiled.options.host_callbacks:
        raise ValueError(
            "A problem compiled with `host_callbacks=True` cannot be saved, because its "
            "programs refer to Python callbacks of this process."
        )
    writer = cyxla.BundleWriter(KIND, VERSION)
    for name in _cache.Structure._fields:
        writer.add(
            name, np.ascontiguousarray(getattr(compiled.structure, name), np.intc)
        )
    records = [
        None
        if executable is None
        else cyxla.pack_program(writer, program.name.lower(), executable)
        for program, executable in zip(
            Program, compiled.native.executables, strict=True
        )
    ]
    header = {
        "xipopt": importlib.metadata.version("xipopt"),
        "cyxla": cyxla.__version__,
        "options": cyxla.encode_record(compiled.options),
        "sense": int(compiled.sense),
        "variables": cyxla.encode_fingerprint(compiled.variables.fingerprint),
        "constraints": cyxla.encode_fingerprint(compiled.constraints.fingerprint),
        "params": cyxla.encode_fingerprint(compiled.params.fingerprint),
        "programs": records,
    }
    return writer, header


def read(
    bundle: cyxla.Bundle,
    variables: PyTree[Any],
    params: PyTree[Any],
    constraints: PyTree[Any],
    variable_bounds: BoundsTree | None,
    on_mismatch: Literal["recompile", "raise"],
    warm_up: bool,
    import_providers: bool,
    workers: int | None,
) -> CompiledProblem:
    """Builds a compiled problem from a bundle and the trees of its variables and parameters."""
    start = time.perf_counter()
    meta = bundle.meta
    records = meta["programs"]
    present = [record for record in records if record]
    needed = cyxla.record_custom_calls(present)
    cyxla.initialize(platform="cpu", jax_targets=bool(needed))
    cyxla.check_targets(present, import_providers)

    variable_layout = layout_for(
        variables, cyxla.decode_fingerprint(meta["variables"]), "variables", is_variable
    )
    expected_constraints = cyxla.decode_fingerprint(meta["constraints"])
    if constraints is None and expected_constraints:
        raise ValueError(
            "The problem in the file has constraints, so `constraints` is required."
        )
    constraint_layout = layout_for(
        () if constraints is None else constraints,
        expected_constraints,
        "constraints",
    )
    lower, upper = bounds_vectors(
        variable_bounds, variable_layout, "variable_bounds", "variables"
    )
    point = start_point(lower, upper)
    layout = params_layout(params)
    cyxla.check_fingerprint(
        cyxla.decode_fingerprint(meta["params"]), layout.fingerprint, "parameters"
    )
    structure = _cache.Structure(
        *(bundle.array(name, np.intc).copy() for name in _cache.Structure._fields)
    )

    loaded_programs = load_programs(bundle, records, on_mismatch, workers)
    statistics = cyxla.CompileStatistics()
    executables = []
    for program, loaded in zip(Program, loaded_programs, strict=True):
        if loaded is None:
            executables.append(None)
            continue
        executable, program_statistics = loaded
        if warm_up:
            warm_up_program(program, executable, point, layout, params)
        statistics = statistics.add(program_statistics)
        executables.append(executable)

    options = cyxla.decode_record(CompileOptions, meta["options"])
    check_output_sizes(
        executables, variable_layout.size, constraint_layout.size, structure
    )
    return CompiledProblem(
        problem=None,
        options=options,
        sense=Sense(meta["sense"]),
        variables=variable_layout,
        params=layout,
        constraints=constraint_layout,
        native=make_native(
            executables,
            variable_layout,
            layout,
            constraint_layout,
            structure,
            options,
        ),
        structure=structure,
        statistics=_cache.combine(statistics, 0.0, 0.0, 0, time.perf_counter() - start),
    )


def load_programs(
    bundle: cyxla.Bundle,
    records: Sequence[dict[str, Any] | None],
    on_mismatch: Literal["recompile", "raise"],
    workers: int | None,
) -> list[tuple[cyxla.Executable, cyxla.CompileStatistics] | None]:
    """Loads the executables of the records, on several threads when there is more than one.

    Deserializing an executable releases the GIL and takes most of the time of a load, so
    the programs overlap. A record that is `None` stays `None`. The first failure is raised
    once every thread has finished. `workers` is the number of threads, which by default is
    one per program up to the number of processors, and 1 loads them one after another.
    """
    present = [record for record in records if record is not None]
    count = min(len(present), os.cpu_count() or 1) if workers is None else workers
    if count <= 1 or len(present) <= 1:
        results = [
            cyxla.load_program(bundle, record, on_mismatch) for record in present
        ]
    else:
        with ThreadPoolExecutor(count) as pool:
            results = list(
                pool.map(
                    lambda record: cyxla.load_program(bundle, record, on_mismatch),
                    present,
                )
            )
    iterator = iter(results)
    return [None if record is None else next(iterator) for record in records]


def warm_up_parameters(
    layout: ParamsLayout, params: PyTree[Any]
) -> Float[np.ndarray, " p"]:
    """Returns the flat parameters to warm up with, which are zeros for shapes without values."""
    if any(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(params)):
        return np.zeros(layout.size)
    return layout.flatten(params)


def warm_up_program(
    program: Program,
    executable: cyxla.Executable,
    point: Float[np.ndarray, " n"],
    layout: ParamsLayout,
    params: PyTree[Any],
) -> None:
    """Runs a program once on a plausible input, so that its first real call is not slower.

    Zeros are a bad input for a program that factorizes a matrix, which fails on a singular
    one. A failure here says nothing about whether the program works elsewhere, since the
    run only loads code, so it raises a warning and loading continues.
    """
    expected = executable.input_types
    assert expected is not None
    values: Sequence[np.ndarray] = [
        point,
        warm_up_parameters(layout, params),
        np.ones(()),
        np.zeros(expected[-1].shape),
    ]
    try:
        executable.warm_up(values[: len(expected)])
    except RuntimeError as error:
        warnings.warn(
            f"The {program.name.lower()} program failed to run once while loading, so it "
            f"was not warmed up: {error}",
            RuntimeWarning,
            stacklevel=4,
        )
