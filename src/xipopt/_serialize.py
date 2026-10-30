"""Saving a compiled problem to a file and loading it in another process.

The file holds the compiled programs, the sparsity structure, the bounds and the options. It
does not hold the functions of the problem or the trees of its variables and parameters. The
caller gives the trees when loading, which makes a loaded problem work exactly like a
compiled one, and the file only keeps the path and shape of each leaf to check them against.
"""

import importlib.metadata
import os
import time
from typing import Any, Literal

import cyxla
import numpy as np
from jaxtyping import PyTree

from xipopt import _cache
from xipopt._compile import CompiledProblem, check_output_sizes, make_native
from xipopt._layout import params_layout, variable_layout_for
from xipopt._options import CompileOptions
from xipopt._types import Program, Sense

KIND = "xipopt.problem"
"""The kind of bundle that holds a compiled problem."""

VERSION = 1
"""The version of the payload, which changes when the data in the file changes."""


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
    on_mismatch: Literal["recompile", "raise"] = "recompile",
    warm_up: bool = True,
) -> CompiledProblem:
    """Returns the compiled problem that `dumps` made from bytes.

    `variables` and `params` are trees with the structure that the problem was compiled for.
    The leaves of `variables` are `Variable`s or arrays, and those of `params` are arrays or
    `jax.ShapeDtypeStruct`s, as for `compile`. Only their paths and shapes are checked
    against the file, and the bounds come from the file. The result solves like a compiled
    problem, with the structure of `variables` for `x`, except that its `problem` is `None`.

    Executables that were stored for another machine are replaced by a compilation of the
    StableHLO, unless `on_mismatch` is `raise`. With `warm_up` each program runs once, so the
    first solve is as fast as the later ones.
    """
    return read(
        cyxla.Bundle(data, KIND, VERSION), variables, params, on_mismatch, warm_up
    )


def load(
    path: str | os.PathLike[str],
    variables: PyTree[Any],
    params: PyTree[Any] = None,
    *,
    on_mismatch: Literal["recompile", "raise"] = "recompile",
    warm_up: bool = True,
) -> CompiledProblem:
    """Returns the compiled problem that `save` wrote to a file, as `loads` does."""
    return read(
        cyxla.Bundle.read(path, KIND, VERSION), variables, params, on_mismatch, warm_up
    )


def pack(compiled: CompiledProblem) -> tuple[cyxla.BundleWriter, dict[str, Any]]:
    """Returns the bundle writer and the header for a compiled problem.

    The programs are added to the writer, and their records go into the header.
    """
    writer = cyxla.BundleWriter(KIND, VERSION)
    arrays = {
        "variable_lower": compiled.variables.lower,
        "variable_upper": compiled.variables.upper,
        "constraint_lower": compiled.constraint_lower,
        "constraint_upper": compiled.constraint_upper,
    }
    for name, values in arrays.items():
        writer.add(name, np.ascontiguousarray(values, np.float64))
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
        "constraint_shapes": [list(shape) for shape in compiled.constraint_shapes],
        "variables": cyxla.encode_fingerprint(compiled.variables.fingerprint),
        "params": cyxla.encode_fingerprint(compiled.params.fingerprint),
        "programs": records,
    }
    return writer, header


def read(
    bundle: cyxla.Bundle,
    variables: PyTree[Any],
    params: PyTree[Any],
    on_mismatch: Literal["recompile", "raise"],
    warm_up: bool,
) -> CompiledProblem:
    """Builds a compiled problem from a bundle and the trees of its variables and parameters."""
    start = time.perf_counter()
    meta = bundle.meta
    records = meta["programs"]
    needed = cyxla.record_custom_calls([record for record in records if record])
    cyxla.initialize(platform="cpu", jax_targets=bool(needed))

    variable_layout = variable_layout_for(
        variables,
        bundle.array("variable_lower", np.float64).copy(),
        bundle.array("variable_upper", np.float64).copy(),
        cyxla.decode_fingerprint(meta["variables"]),
    )
    layout = params_layout(params)
    cyxla.check_fingerprint(
        cyxla.decode_fingerprint(meta["params"]), layout.fingerprint, "parameters"
    )
    structure = _cache.Structure(
        *(bundle.array(name, np.intc).copy() for name in _cache.Structure._fields)
    )

    statistics = cyxla.CompileStatistics()
    executables = []
    for record in records:
        if record is None:
            executables.append(None)
            continue
        executable, loaded = cyxla.load_program(bundle, record, on_mismatch)
        if warm_up:
            executable.warm_up()
        statistics = statistics.add(loaded)
        executables.append(executable)

    options = cyxla.decode_record(CompileOptions, meta["options"])
    constraint_lower = bundle.array("constraint_lower", np.float64).copy()
    constraint_upper = bundle.array("constraint_upper", np.float64).copy()
    check_output_sizes(
        executables, variable_layout.size, constraint_lower.size, structure
    )
    return CompiledProblem(
        problem=None,
        options=options,
        sense=Sense(meta["sense"]),
        variables=variable_layout,
        params=layout,
        constraint_shapes=tuple(tuple(shape) for shape in meta["constraint_shapes"]),
        constraint_lower=constraint_lower,
        constraint_upper=constraint_upper,
        native=make_native(
            executables,
            variable_layout,
            layout,
            constraint_lower,
            constraint_upper,
            structure,
            options,
        ),
        structure=structure,
        statistics=_cache.combine(statistics, 0.0, 0.0, 0, time.perf_counter() - start),
    )
