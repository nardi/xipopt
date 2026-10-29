# Installation

```bash
pip install xipopt
```

xipopt needs Python 3.11 or newer. Wheels exist for Linux (x86-64 and arm64), macOS
(Intel and Apple silicon) and Windows (x86-64). They contain Ipopt 3.14 with the sparse
linear solvers MUMPS and SPRAL, OpenBLAS and the runtime libraries of the compiler, so
the system needs no Ipopt or BLAS. [cyxla](https://nardi.github.io/cyxla/) and its CPU
plugin come as dependencies, which means nothing downloads at run time.

## Python versions

The sparsity detection of [asdex](https://pypi.org/project/asdex/) needs Python 3.12 or
newer, and so does JAX 0.11. On Python 3.11 pip installs JAX 0.10 and leaves out asdex.
xipopt then computes dense derivatives and warns about it, unless the problem gives the
patterns of its Jacobian and Hessian. The [derivatives guide](guide/derivatives.md)
describes the choices.

## JAX versions

xipopt works with JAX 0.10 and 0.11. The version of asdex that the detection relies on
does not import on JAX 0.12 yet, so the dependency stops below it. The PJRT plugin of
cyxla is a build of XLA 0.12 and compiles the StableHLO that these versions of JAX
export.

## Licenses

xipopt itself is under the Hippocratic License 3.0. The wheels also contain Ipopt
(Eclipse Public License 2.0), MUMPS (CeCILL-C), SPRAL, METIS, OpenBLAS and a few smaller
libraries, each under its own license. Every wheel carries a copy of them in its
metadata, and the [licenses
directory](https://github.com/nardi/xipopt/tree/main/licenses) of the repository lists
them.

## Building from source

The extension module links the Ipopt library of the
[Ipopt_jll](https://github.com/JuliaBinaryWrappers/Ipopt_jll.jl) package that Julia
publishes. [meson-jll](https://nardi.github.io/meson-jll/) turns it into Meson wraps.
The repository commits only the lock file `subprojects/meson-jll.lock.wrap`, and the
first configuration downloads `meson-jll` from its release and writes the wraps. You
need a C compiler, and Meson fetches the rest.

```bash
git clone https://github.com/nardi/xipopt
cd xipopt
uv sync
```

The [development page](development.md) describes the checks that the repository runs.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `CYXLA_PJRT_PLUGIN` | Path of a PJRT plugin that replaces the installed one |
| `CYXLA_CACHE_DIR` | Directory of the disk cache, `~/.cache/cyxla` if unset |
| `OMP_CANCELLATION` | Set to `TRUE` by xipopt unless the environment says otherwise, since SPRAL needs it |
