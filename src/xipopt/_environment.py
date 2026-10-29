"""Prepares the process for the native libraries before the extension module loads.

Linux and macOS find the bundled libraries through the run path that meson-python writes
into the extension module. Windows has no run path, so the directory of the libraries is
added to the search path of DLLs. SPRAL needs OpenMP cancellation, which the OpenMP runtime
reads from the environment when it loads.
"""

import os
import sys
from pathlib import Path

LIBRARY_DIRECTORY = Path(__file__).resolve().parent.parent / ".xipopt.mesonpy.libs"
"""Where meson-python puts the libraries of a wheel, next to the package."""

if sys.platform == "win32":
    if LIBRARY_DIRECTORY.is_dir():
        os.add_dll_directory(str(LIBRARY_DIRECTORY))
    else:
        # An editable install leaves the libraries in the build tree, one directory per package.
        build = Path(__file__).resolve().parents[2] / "build"
        for directory in {path.parent for path in build.glob("*/subprojects/*/*.dll")}:
            os.add_dll_directory(str(directory))

os.environ.setdefault("OMP_CANCELLATION", "TRUE")


def library_path(name: str) -> bytes | None:
    """Returns the path of a loaded library, or `None` if it is unknown.

    The native layer asks for this on Windows only. The address of an imported function
    there belongs to a thunk of the extension module and not to the library, so the library
    is looked up by its file name instead.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel.GetModuleHandleW.restype = wintypes.HMODULE
        kernel.GetModuleFileNameW.argtypes = [
            wintypes.HMODULE,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        module = kernel.GetModuleHandleW(name)
        if not module:
            return None
        buffer = ctypes.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(module, buffer, len(buffer)):
            return None
        return os.fsencode(buffer.value)
    return None  # pragma: no cover
