python.install_sources(
  '__init__.py',
  '_cache.py',
  '_compile.py',
  '_layout.py',
  '_libraries.py',
  '_options.py',
  '_patterns.py',
  '_problem_types.py',
  '_programs.py',
  '_solve.py',
  '_types.py',
  '_solver.pyi',
  'py.typed',
  subdir: 'xipopt',
)

cython_args = ['-3', '--fast-fail']
# MSVC knows none of these options, so only the supported ones are passed.
c_args = meson.get_compiler('c').get_supported_arguments(
  '-Wno-unused-function', '-Wno-unused-variable', '-Wno-unused-but-set-variable',
)
if get_option('coverage')
  cython_args += ['-X', 'linetrace=True']
  # Coverage measures branches through `sys.settrace` before Python 3.14, so Cython has to
  # report lines through the same hooks rather than through `sys.monitoring`.
  c_args += ['-DCYTHON_TRACE_NOGIL=1', '-DCYTHON_USE_SYS_MONITORING=0']
endif

# Meson compiles Cython from `.pyx` sources only, so the pure Python mode module is
# translated to C by an explicit `cython` step. The declarations are listed as dependencies,
# because the depfile would record paths relative to the project root rather than the build
# directory.
solver_c = custom_target(
  '_solver.c',
  input: '_solver.py',
  output: '_solver.c',
  depend_files: files('_blas.pxd', '_ipopt.pxd', '_platform.h', '_platform.pxd', '_solver.pxd'),
  command: [
    cython, cython_args,
    # Running from the project root makes Cython record the source as
    # `src/xipopt/_solver.py`, the path coverage resolves line events against.
    '-w', meson.project_source_root(),
    '-I', 'src',
    '-I', cyxla_include[1],
    '--module-name', 'xipopt._solver',
    meson.current_source_dir() / '_solver.py',
    '-o', meson.current_build_dir() / '_solver.c',
  ],
)
if get_option('coverage')
  # The `Cython.Coverage` plugin looks for the generated C file next to the source, so
  # coverage builds place a copy there.
  custom_target(
    '_solver.c.copy',
    input: solver_c,
    output: '_solver.c.stamp',
    command: [
      python, '-c', 'import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2]); open(sys.argv[3], "w").close()',
      '@INPUT@', meson.current_source_dir() / '_solver.c', '@OUTPUT@',
    ],
    build_by_default: true,
  )
endif
python.extension_module(
  '_solver',
  solver_c,
  dependencies: [ipopt, openblas, lbt, cyxla, python.dependency()],
  # `_platform.h` sits next to the sources and is not installed, as only this module needs it.
  include_directories: include_directories('.'),
  c_args: c_args,
  install: true,
  subdir: 'xipopt',
)
