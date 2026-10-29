Use match statements with unpacking where it makes sense (instead of if-else chains).
Try to avoid inheritance where possible, prefer defining Protocols and separate implementations (structural typing). You can reuse behaviors by defining functions.
Make sure all code is properly typed and typechecks with ty.
For Cython code, use the pure Python interface, and add accurate type stubs for exported functions.
Callbacks that Ipopt calls run without the GIL, so they must not allocate Python objects or raise. The only exception is the iteration observer, which takes the GIL.
The programs run in 64-bit floats. Lower and trace inside `jax.enable_x64(True)` and never rely on the global setting of the caller.
Bump the pin on `cyxla` in `pyproject.toml` whenever `ABI_VERSION` in `cyxla` changes.
Keep the runtime portable to Linux, macOS and Windows. Operating system calls go through `src/xipopt/_platform.h`.
