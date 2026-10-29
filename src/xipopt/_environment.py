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

os.environ.setdefault("OMP_CANCELLATION", "TRUE")


def library_path(address: int) -> bytes | None:
    """Returns the path of the library that holds an address, or `None` if it is unknown.

    The native layer asks for this on Windows only, where C code has no portable way to do it.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleExW.argtypes = [
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.HMODULE),
        ]
        kernel.GetModuleFileNameW.argtypes = [
            wintypes.HMODULE,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        module = wintypes.HMODULE()
        # The flags look the module up by an address in it, without counting a reference.
        flags = 0x4 | 0x2
        if not kernel.GetModuleHandleExW(flags, address, ctypes.byref(module)):
            return None
        buffer = ctypes.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(module, buffer, len(buffer)):
            return None
        return os.fsencode(buffer.value)
    return None  # pragma: no cover
